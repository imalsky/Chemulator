#!/bin/bash
#SBATCH -J MYGPUJOB
#SBATCH -o MYGPUJOB.o%j
#SBATCH -e MYGPUJOB.e%j

# ---- MIG partition (ONLY keep gpu-mig for MIG jobs) ----
#SBATCH -p gpu-mig
#SBATCH --gres=gpu:2g.20gb:1

# CPU/RAM/time (MIG jobs typically map 1 GPU TRES to 8 CPUs on your system)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH -t 5:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

# -------- toggles (override via: sbatch --export=ALL,DEBUG=1,SAN=1,CONDA_ENV=nn,SKIP_PIP=1) --------
DEBUG=${DEBUG:-0}
SAN=${SAN:-0}
CONDA_ENV=${CONDA_ENV:-nn}
SKIP_PIP=${SKIP_PIP:-0}     # 1 = do not pip install into vendor (assume env is ready)
PIP_TIMEOUT_SEC=${PIP_TIMEOUT_SEC:-900}  # fail pip if it takes > 15 minutes

if [ "$DEBUG" -eq 1 ]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_SHOW_CPP_STACKTRACES=1
  export PYTHONFAULTHANDLER=1
  export CUDA_MODULE_LOADING=EAGER
  export CUDA_CACHE_DISABLE=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export NVIDIA_TF32_OVERRIDE=0
fi

RUNNER=()
if [ "$SAN" -eq 1 ]; then
  RUNNER=(compute-sanitizer --tool memcheck --leak-check full --report-api-errors yes --show-backtrace yes --error-exitcode=99)
fi

# -------- project root (PROJECT_ROOT > SLURM_SUBMIT_DIR > script dir) --------
ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}}"
cd -P "$ROOT"
export ROOT="$(/bin/pwd -P)"

# -------- mirror stdout/stderr to a logfile AND SLURM output --------
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/run_${RUN_ID}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Project root: $ROOT"
echo "Log file: $LOG_FILE"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "Batch host: ${SLURM_JOB_NODELIST:-unknown}"

# -------- conda env --------
CONDA_EXE=$(command -v conda || true)
[ -n "$CONDA_EXE" ] || { echo "conda not found"; exit 2; }

CONDA_BASE=$(dirname "$(dirname "$CONDA_EXE")")
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "failed to activate env '$CONDA_ENV'"; exit 2; }

PY="$(command -v python)"
echo "Python: $PY"

# -------- optional CUDA module (non-fatal) --------
if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

# -------- stability / FS --------
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
ulimit -c unlimited || true

# -------- JOB-SCOPED vendor dir (prevents version soup + repeated upgrades) --------
VENDOR_BASE="${VENDOR_BASE:-$ROOT/vendor_pkgs}"
VENDOR="$VENDOR_BASE/${SLURM_JOB_ID:-manual}"
mkdir -p "$VENDOR"
echo "Vendor dir: $VENDOR"

# pip opts (no --upgrade; install only if missing/wrong)
PIP_Q_OPTS=(--quiet --no-input --disable-pip-version-check --no-color --no-cache-dir --target "$VENDOR" --no-warn-script-location)

SITE_PKGS="$("$PY" -c "import site; c=[p for p in site.getsitepackages() if p.endswith('site-packages')]; print(c[0] if c else site.getsitepackages()[0])")"

# vendor FIRST so it can override env when needed
export PYTHONPATH="$VENDOR:$SITE_PKGS:$ROOT:$ROOT/src:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# -------- make config visible if code expects src/config/config.json --------
if [ -f "$ROOT/config/config.json" ] && [ ! -f "$ROOT/src/config/config.json" ]; then
  mkdir -p "$ROOT/src/config"
  ln -sf "$ROOT/config/config.json" "$ROOT/src/config/config.json"
  echo "Linked $ROOT/src/config/config.json -> $ROOT/config/config.json"
fi

pip_install() {
  # usage: pip_install pkg1 pkg2 ...
  echo "pip install: $*"
  timeout "$PIP_TIMEOUT_SEC" "$PY" -m pip install "${PIP_Q_OPTS[@]}" "$@"
}

# ===============================
# Ensure critical Python deps
# ===============================
if [ "$SKIP_PIP" -ne 1 ]; then
  # numpy: enforce <2
  if ! "$PY" - <<'PY' >/dev/null 2>&1
import numpy as np
raise SystemExit(0 if int(np.__version__.split('.')[0]) < 2 else 1)
PY
  then
    pip_install "numpy==1.26.4"
  fi

  # scipy: enforce >=1.14,<2
  if ! "$PY" - <<'PY' >/dev/null 2>&1
import scipy
maj, minr = map(int, scipy.__version__.split('.')[:2])
raise SystemExit(0 if (maj == 1 and minr >= 14) else 1)
PY
  then
    pip_install "scipy>=1.14,<2"
  fi

  # lightning namespace (your code imports lightning.pytorch)
  if ! "$PY" -c "import lightning.pytorch as pl; print(pl.__version__)" >/dev/null 2>&1; then
    pip_install "lightning>=2.0,<3" "tensorboard>=2.13,<3"
  fi

  # optional: legacy import coverage
  if ! "$PY" -c "import pytorch_lightning as pl; print(pl.__version__)" >/dev/null 2>&1; then
    pip_install "pytorch-lightning>=2.0,<3"
  fi

  # torch-optimizer (LAMB)
  if ! "$PY" -c "import torch_optimizer" >/dev/null 2>&1; then
    pip_install "torch-optimizer>=0.3.0,<0.5"
  fi

  # h5py / optuna only if you actually need them
  if ! "$PY" -c "import h5py" >/dev/null 2>&1; then
    pip_install "h5py>=3.9,<3.15"
  fi
  if ! "$PY" -c "import optuna" >/dev/null 2>&1; then
    pip_install "optuna>=4.4.0,<5"
  fi
else
  echo "SKIP_PIP=1; skipping vendor installs."
fi

echo "Dependency versions (effective imports with current PYTHONPATH):"
"$PY" - <<'PY'
import numpy, scipy
import lightning.pytorch as pl
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("lightning", pl.__version__)
PY

echo "CUDA visibility (inside batch):"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

# -------- sanity checks --------
[ -f "$ROOT/src/main.py" ] || { echo "Missing $ROOT/src/main.py"; exit 1; }
[ -f "$ROOT/config/config.json" ] || [ -f "$ROOT/src/config/config.json" ] || { echo "Missing config.json under config/ or src/config/"; exit 1; }

# ===============================
# Train
# ===============================
echo "Starting training..."
cd "$ROOT"
exec /usr/bin/env time -v "${RUNNER[@]}" "$PY" -u "$ROOT/src/main.py"
