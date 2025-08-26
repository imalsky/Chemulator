#!/usr/bin/env python3
"""Plot AE-DeepONet predictions vs ground truth on the shard’s full (dense) time grid."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="1"

from pathlib import Path; import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np, torch
torch.set_num_threads(1)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_json, seed_everything
from normalizer import NormalizationHelper
from model import AEDeepONet

# --- CONFIG ---
MODEL_PATH  = Path("../models/ae_deeponet_20250825_153522/best_model.pt")
CONFIG_PATH = Path("../models/ae_deeponet_20250825_153522/config.json")
DATA_DIR    = Path("../data/processed")
SAMPLE_INDEX = 5
OUTPUT_DIR   = None
SEED = 42
# --------------

def load_model(path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        m = AEDeepONet(cfg).to(device); m.load_state_dict(obj["model_state_dict"], strict=False)
    elif hasattr(obj, "eval"):
        m = obj.to(device) if hasattr(obj, "to") else obj
    else:
        raise RuntimeError(f"Unsupported model file: {path}")
    m.eval(); return m

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_json(CONFIG_PATH)

    data_dir = Path(cfg.get("paths", {}).get("processed_data_dir", DATA_DIR))
    if not data_dir.is_absolute(): data_dir = Path("..") / data_dir
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else MODEL_PATH.parent / "plots"; out_dir.mkdir(parents=True, exist_ok=True)

    norm = NormalizationHelper(load_json(data_dir / "normalization.json"), device, cfg)
    model = load_model(MODEL_PATH, cfg, device)

    # Load a test shard (must contain dense t_vec)
    test_shards = sorted((data_dir / "test").glob("shard_*.npz"))
    if not test_shards: raise RuntimeError(f"No test shards in {data_dir/'test'}")
    with np.load(test_shards[0], allow_pickle=False) as d:
        if "t_vec" not in d.files: raise RuntimeError("Shard missing 't_vec'; cannot evaluate on dense grid.")
        x0_np = d["x0"].astype(np.float32)        # [N,S]
        y_np  = d["y_mat"].astype(np.float32)     # [N,M_dense,S]  (physical space)
        g_np  = d["globals"].astype(np.float32)   # [N,G]
        t_np  = d["t_vec"]                         # [M_dense] or [N,M_dense] (physical time)

    N = x0_np.shape[0]
    idx = SAMPLE_INDEX if SAMPLE_INDEX is not None else np.random.randint(0, N)

    data_cfg    = cfg["data"]
    species     = data_cfg["species_variables"]
    gvars_names = data_cfg["global_variables"]
    t_name      = data_cfg["time_variable"]

    # Slice one sample
    x0    = torch.from_numpy(x0_np[idx:idx+1]).to(device)      # [1,S]
    y_true= y_np[idx]                                          # [M_dense,S] (physical)
    t_vec = t_np[idx] if t_np.ndim == 2 else t_np              # [M_dense]   (physical)

    # Build inputs: z0 + globals (normalized)
    x0_norm  = norm.normalize(x0, species)                     # [1,S]
    z0       = model.encode(x0_norm)                           # [1,L]
    g_t      = torch.from_numpy(g_np[idx:idx+1]).to(device)
    g_norm   = norm.normalize(g_t, gvars_names)                # [1,G]
    branch   = torch.cat([z0, g_norm], dim=-1)                 # [1,L+G]

    # Trunk on the dense time grid (normalize time to the training scale)
    t_dense = torch.tensor(t_vec, dtype=torch.float32, device=device).unsqueeze(1)  # [M_dense,1]
    t_norm  = norm.normalize(t_dense, [t_name]).squeeze(1)                          # [M_dense] in [0,1]

    with torch.no_grad():
        y_pred_norm, _ = model(branch, decode=True, trunk_times=t_norm)             # [1,M_dense,S]
    y_pred = norm.denormalize(y_pred_norm.squeeze(0), species).cpu().numpy()        # [M_dense,S]

    # Plot (log-safe)
    eps = 1e-30
    t_plot    = np.clip(t_vec, eps, None)
    y_true_pl = np.clip(y_true, eps, None)
    y_pred_pl = np.clip(y_pred, eps, None)

    fig, ax = plt.subplots(figsize=(10,7))
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(species)))
    for i, sp in enumerate(species):
        ax.loglog(t_plot, y_true_pl[:, i], '-',  color=colors[i], lw=2.2, alpha=0.9)
        ax.loglog(t_plot, y_pred_pl[:, i], '--', color=colors[i], lw=1.8, alpha=0.85)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0], color='black', lw=2.2, ls='-'),
               Line2D([0],[0], color='black', lw=1.8, ls='--')], ['True','Predicted'],
              loc='lower left', fontsize=11)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Species Abundance")
    ax.set_title(f"AE-DeepONet vs Ground Truth (Sample {idx})")
    ax.grid(True, which="both", alpha=0.3, ls="--")
    plt.tight_layout()
    path = out_dir / f"predictions_dense_sample_{idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved to {path}")

    rel = np.abs(y_pred_pl - y_true_pl) / (np.abs(y_true_pl) + 1e-10)
    for i, sp in enumerate(species): print(f"{sp:15s}: mean rel err = {rel[:, i].mean():.3e}")
    print(f"\n{'Overall':15s}: mean = {rel.mean():.3e}, max = {rel.max():.3e}")

if __name__ == "__main__":
    main()
