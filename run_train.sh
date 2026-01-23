#!/bin/bash
#SBATCH -J flowmap
#SBATCH -o flowmap.o%j
#SBATCH -e flowmap.e%j
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --clusters=edge
#SBATCH --cpus-per-gpu=32
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH -t 5:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

set -euo pipefail

# ==============================================================================
# Configuration (override via: sbatch --export=ALL,CONDA_ENV=myenv)
# ==============================================================================

CONDA_ENV="${CONDA_ENV:-nn}"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

# ==============================================================================
# Environment Setup
# ==============================================================================

cd -P "$PROJECT_ROOT"
ROOT="$(/bin/pwd -P)"

# Activate conda
CONDA_BASE="$(conda info --base 2>/dev/null)" || { echo "conda not found"; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda env '$CONDA_ENV'"; exit 1; }

# Load CUDA module if available
if command -v module &>/dev/null; then
    module load cuda12.6/toolkit 2>/dev/null || \
    module load cuda11.8/toolkit 2>/dev/null || true
fi

# Stability settings
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Add project to Python path
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

# ==============================================================================
# Run Training
# ==============================================================================

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS"
echo "Conda env:    $CONDA_ENV"
echo "Project root: $ROOT"
echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $(python -c 'import torch; print(torch.version.cuda or "N/A")')"
echo "=========================================="

cd "$ROOT/src"
python -u main.py

echo "Training complete."