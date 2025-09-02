#!/bin/bash
#SBATCH -J GosJob               # Job name
#SBATCH -o GosJob.o%j            # Standard output file
#SBATCH -e GosJob.e%j            # Standard error file
#SBATCH -p gpu                          # Specify the GPU partition
#SBATCH --cpus-per-task=4              # Request 32 CPU cores
#SBATCH -A exoweather



#SBATCH --clusters=edge                 # Target the edge nodes
#SBATCH -N 1                            # Request a single node
#SBATCH -n 1                            # Run a single task
#SBATCH --gpus=1                        # Request one full GPU
#SBATCH --cpus-per-task=16              # Request 32 CPU cores


#SBATCH --mem=100G                      # Request 100 GB of CPU RAM

####SBATCH -p gpu-mig
####SBATCH --gres=gpu:2g.20gb:1


#SBATCH -t 8:00:00                     # Set a 24-hour runtime limit

cd "$SLURM_SUBMIT_DIR"

# Activate Conda environment
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting." >&2; exit 1;
fi

CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'." >&2; exit 1; }

# Try to load any available CUDA module
echo "Looking for available CUDA modules..."
module avail cuda 2>&1 | grep -i cuda || echo "No CUDA modules found via module system"

# Try different common CUDA versions
for cuda_version in cuda/12.7 cuda/12.6 cuda/12.5 cuda/12.4 cuda/12.0 cuda/11.8 cuda/11.7 cuda; do
    if module load "$cuda_version" 2>/dev/null; then
        echo "Successfully loaded $cuda_version"
        break
    fi
done
# --- Run the Application ---
echo "Starting Python application..."
python src/main.py
echo "Job completed successfully."