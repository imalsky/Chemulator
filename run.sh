#!/bin/bash
#SBATCH -J ChemulatorJob                # Job name
#SBATCH -o ChemulatorJob.o%j            # Standard output file
#SBATCH -e ChemulatorJob.e%j            # Standard error file
#SBATCH -p gpu                          # Specify the GPU partition
#SBATCH --clusters=edge                 # Target the edge nodes
#SBATCH -N 1                            # Request a single node
#SBATCH -n 1                            # Run a single task
#SBATCH --gpus=1                        # Request one full GPU
#SBATCH --cpus-per-task=32              # Request 32 CPU cores
#SBATCH --mem=150G                      # Request 100 GB of CPU RAM
#SBATCH -t 72:00:00                     # Set a 24-hour runtime limit

cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting." >&2; exit 1;
fi
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'." >&2; exit 1; }

# Load CUDA module
module load cuda/11.8 2>/dev/null || echo "Warning: Failed to load CUDA module."

# --- Print Job Configuration to Log File ---
echo "------------------------------------------------"
echo "JOB CONFIGURATION"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CPU cores requested: $SLURM_CPUS_PER_TASK"
nvidia-smi
echo "------------------------------------------------"

# --- Run the Application ---
echo "Starting Python application..."
python src/main.py --tune
echo "Job completed successfully."