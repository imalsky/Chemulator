import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

plt.style.use('science.mplstyle')

# --- THE FIX ---
# Force matplotlib to use the regular text font for math (exponents)
plt.rcParams.update({'mathtext.default':  'regular'})

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "big_big_big"
CSVS = MODEL_DIR / "csv"
PLOT_DIR = MODEL_DIR / "plots"

# 1. Load & Merge
files = sorted(CSVS.glob("version_*/metrics.csv"), key=lambda p: int(p.parent.name.split('_')[1]))
dfs = [pd.read_csv(p).groupby('epoch').last().reset_index() for p in files]
df = pd.concat(dfs, ignore_index=True)

# 2. Fix gaps
df = df.dropna(subset=['train_loss']).reset_index(drop=True)
df['epoch'] = df.index

# 3. Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(df['epoch'], df['train_loss'], label='Train')

ax.set(xlabel='Epoch', ylabel='Loss', yscale='log', box_aspect=1)
ax.set_xlim(df['epoch'].min(), df['epoch'].max())

# 4. Parallel Axis (Steps)
ax2 = ax.twiny()
ax2.set_xlim(df['step'].min(), df['step'].max())
ax2.set_xlabel('Step')

def format_sci(x, pos):
    if x == 0: return "0"
    s = f'{x:.1e}'
    base, exponent = s.split('e')
    # Because of the rcParams update above, this now uses the regular font
    return r'${} \times 10^{{{}}}$'.format(base, int(exponent))

ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_sci))

PLOT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOT_DIR / "training.png", dpi=150, bbox_inches="tight")
plt.close(fig)