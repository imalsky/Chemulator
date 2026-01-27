#!/bin/bash
#SBATCH -J MYGPUJOB
#SBATCH -o MYGPUJOB.o%j
#SBATCH -e MYGPUJOB.e%j
#SBATCH -p gpu
#SBATCH --clusters=edge
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH -t 5:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-nn}

cd -P "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"

CONDA_EXE="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# optional CUDA module (ignore failures)
if command -v module >/dev/null 2>&1; then
  module load cuda12.6/toolkit 2>/dev/null || module load cuda11.8/toolkit 2>/dev/null || true
fi

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"

echo "Running: python -u src/model.py"
srun python -u src/main.py
