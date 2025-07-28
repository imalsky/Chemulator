# testing/plot_losses.py
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent

def get_latest_model_dir():
    model_path = ROOT / 'data' / 'models'
    dirs = [d for d in model_path.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_ctime) if dirs else None

MODEL_DIR = get_latest_model_dir()

def main():
    if not MODEL_DIR:
        raise ValueError("No model directory found")
    
    log_path = MODEL_DIR / 'training_log.json'
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found in {MODEL_DIR}")
    
    history = json.loads(log_path.read_text())
    epochs = history['epochs']
    
    epoch_nums = [e['epoch'] for e in epochs]
    train_losses = [e['train_loss'] for e in epochs]
    val_losses = [e['val_loss'] for e in epochs if 'val_loss' in e]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_nums, train_losses, 'b-', label='Train Loss')
    if val_losses:
        plt.plot(epoch_nums[:len(val_losses)], val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plots_dir = MODEL_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / 'loss_curve.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()