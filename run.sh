#!/bin/bash
#SBATCH -J GosJob               # Job name
#SBATCH -o GosJob.o%j           # Standard output file
#SBATCH -e GosJob.e%j           # Standard error file
#SBATCH -p gpu                  # Specify the GPU partition
#SBATCH --cpus-per-task=4       # Request 4 CPU cores
#SBATCH -A exoweather

#SBATCH --clusters=edge         # Target the edge nodes
#SBATCH -N 1                    # Request a single node
#SBATCH -n 1                    # Run a single task
#SBATCH --gpus=1                # Request one full GPU
#SBATCH --cpus-per-task=16      # Request 16 CPU cores

#SBATCH --mem=100G              # Request 100 GB of CPU RAM
#SBATCH -t 8:00:00              # Set an 8-hour runtime limit

cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting." >&2
    exit 1
fi

CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'." >&2; exit 1; }

# Method 1: Check available modules first, then load
# This completely avoids trying to load non-existent modules
echo "Checking for available CUDA modules..."
CUDA_LOADED=false

# Get list of available CUDA modules (suppress all output)
available_cuda=$(module avail cuda 2>&1 | grep -o 'cuda/[0-9.]*\|cuda ' | head -n 1)

if [ -n "$available_cuda" ]; then
    # Try to load the first available CUDA module
    if module load $available_cuda &>/dev/null; then
        echo "CUDA module loaded: $available_cuda"
        CUDA_LOADED=true
    fi
fi

# --- Run the Application ---
python src/main.py
echo "Job completed successfully."