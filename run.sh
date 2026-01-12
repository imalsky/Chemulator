#!/bin/bash
#SBATCH -J MYGPUJOB         # Job name
#SBATCH -o MYGPUJOB.o%j     # Name of job output file
#SBATCH -e MYGPUJOB.e%j     # Name of stderr error file
#SBATCH -p gpu              # Queue (partition) name for GPU nodes
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH -n 16                # Total # of CPU cores (adjust as needed)
#SBATCH --clusters=edge     # *** CRITICAL: Directs the job to the edge cluster nodes (gn11-gn14) ***
#SBATCH --cpus-per-gpu=32
#SBATCH --gpus=1            # Request 1 GPU (adjust if more needed)
#SBATCH --mem=50G           # Memory (RAM) requested for gpu-mig
#SBATCH -t 5:00:00         # Run time (hh:mm:ss) for gpu-mig (adjust if needed)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Minimal, quiet SLURM launcher (conda + optional CUDA + vendored deps + preprocess + train)

set -euo pipefail

# -------- toggles (override via: sbatch --export=ALL,DEBUG=1,SAN=1,CONDA_ENV=nn) --------
DEBUG=${DEBUG:-0}     # 1 = sync kernels, FP32, tiny run
SAN=${SAN:-0}        # 1 = compute-sanitizer wrap (slow)
CONDA_ENV=${CONDA_ENV:-nn}

if [ "$DEBUG" -eq 1 ]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_SHOW_CPP_STACKTRACES=1
  export PYTHONFAULTHANDLER=1
  export CUDA_MODULE_LOADING=EAGER
  export CUDA_CACHE_DISABLE=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export NVIDIA_TF32_OVERRIDE=0
  PRECISION_ARG="--precision 32-true"
  LIMIT_ARGS="--max_steps 1 --limit_train_batches 1 --limit_val_batches 0 --num_sanity_val_steps 0"
else
  PRECISION_ARG=""
  LIMIT_ARGS=""
fi

RUNNER=()
if [ "$SAN" -eq 1 ]; then
  RUNNER=(compute-sanitizer --tool memcheck --leak-check full --report-api-errors yes --show-backtrace yes --error-exitcode=99)
fi

# -------- project root --------
cd -P "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export ROOT="$(/bin/pwd -P)"

# -------- conda env --------
CONDA_EXE=$(command -v conda || true); [ -n "$CONDA_EXE" ] || { echo "conda not found"; exit 2; }
CONDA_BASE=$(dirname "$(dirname "$CONDA_EXE")")
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "failed to activate env '$CONDA_ENV'"; exit 2; }
PY="$(command -v python)"

# -------- optional CUDA module (non-fatal) --------
if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

# -------- stability / FS --------
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
ulimit -c unlimited || true

# -------- vendor dir (quiet installs) --------
if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ]; then
  VENDOR="${GLOBAL_VENDOR_BASE:-$SCRATCH/vendor_pkgs}"
else
  VENDOR="${GLOBAL_VENDOR_BASE:-$HOME/vendor_pkgs}"
fi
mkdir -p "$VENDOR"

PIP_Q_OPTS=(--quiet --no-input --disable-pip-version-check --no-color --no-cache-dir --upgrade --target "$VENDOR" --no-warn-script-location)

# -------- PYTHONPATH order --------
SITE_PKGS="$("$PY" -c "import site; c=[p for p in site.getsitepackages() if p.endswith('site-packages')]; print(c[0] if c else site.getsitepackages()[0])")"
export PYTHONPATH="$SITE_PKGS:$VENDOR:$ROOT:$ROOT/src:${PYTHONPATH:-}"

# ===============================
# Ensure critical Python deps
# ===============================

# numpy
if ! "$PY" -c "import numpy; import importlib; importlib.import_module('numpy.core._multiarray_umath')" >/dev/null 2>&1; then
  "$PY" -m pip install "${PIP_Q_OPTS[@]}" "numpy==1.26.4"
fi

# h5py
if ! "$PY" -c "import h5py" >/dev/null 2>&1; then
  "$PY" -m pip install "${PIP_Q_OPTS[@]}" "h5py>=3.9,<3.15"
fi

# optuna
if ! "$PY" -c "import optuna" >/dev/null 2>&1; then
  "$PY" -m pip install "${PIP_Q_OPTS[@]}" "optuna>=4.4.0,<5"
fi

# pytorch-lightning + tensorboard
if ! "$PY" -c "import pytorch_lightning" >/dev/null 2>&1; then
  "$PY" -m pip install "${PIP_Q_OPTS[@]}" "pytorch-lightning>=2.4,<3" "tensorboard>=2.13,<3"
fi

# torch-optimizer (for LAMB)
if ! "$PY" -c "import torch_optimizer" >/dev/null 2>&1; then
  "$PY" -m pip install "${PIP_Q_OPTS[@]}" "torch-optimizer>=0.3.0,<0.5"
fi


# ===============================
# Phase B: train (single GPU)
# ===============================
echo "Starting training on single GPU..."
/usr/bin/env time -v "${RUNNER[@]}" "$PY" -u model.py $PRECISION_ARG $LIMIT_ARGS
