from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
MODEL_STR = "flowmap-deeponet"
MODEL_DIR = f"../models/{MODEL_STR}"
TRAINING_STR = f"{MODEL_DIR}/training_log.txt"
OUTPUT_DIR = None
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

try:
    plt.style.use("science.mplstyle")
except Exception:
    pass


data = pd.read_csv(TRAINING_STR)

plt.plot(data['epoch'], data['train_loss'])
plt.plot(data['epoch'], data['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE loss')

plt.yscale("log")
plt.tight_layout()
output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(MODEL_DIR) / "plots"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "train.png", dpi=150, bbox_inches="tight")
plt.close()
