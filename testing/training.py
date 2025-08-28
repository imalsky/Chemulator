from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Configuration
MODEL_STR = "deepo"
MODEL_DIR = f"../models/{MODEL_STR}"
CONFIG_PATH = f"{MODEL_DIR}/config.json"
PROCESSED_DIR = "../data/processed-10-log-standard"
TRAINING_STR = f"{MODEL_DIR}/training_history.json"
OUTPUT_DIR = None

# Add parent src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_json, seed_everything

# Load training history and extract MSE errors
history = load_json(Path(TRAINING_STR))
mse_errors = [epoch['mse'] for epoch in history['train']]

# Set up output directory
output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(MODEL_DIR) / "plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "train.png"

# Create and save plot
plt.plot(mse_errors)
plt.yscale("log")
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()