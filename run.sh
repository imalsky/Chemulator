#!/bin/bash
#SBATCH -J ChemulatorJob                # Job name
#SBATCH -o ChemulatorJob.o%j            # Standard output file
#SBATCH -e ChemulatorJob.e%j            # Standard error file
#SBATCH -p gpu                          # Specify the GPU partition
#SBATCH --clusters=edge                 # Target the edge nodes
#SBATCH -N 1                            # Request a single node
#SBATCH -n 1                            # Run a single task
#SBATCH --gpus=1                        # Request one full GPU
#SBATCH --cpus-per-task=16              # Request 32 CPU cores
#SBATCH --mem=240G                      # Request 240 GB of CPU RAM
#SBATCH -t 24:00:00                     # Set a 48-hour runtime limit (for tuning)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Change to the directory from which the script was submitted
cd "$SLURM_SUBMIT_DIR"

# Dynamically locate Conda initialization script
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting."
    exit 1
fi

CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate your Conda environment
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'. Exiting."; exit 1; }

# Check if the 'module' command is available
if command -v module &> /dev/null; then
    # Initialize module system
    source /usr/share/Modules/init/bash 2>/dev/null || \
    source /etc/profile.d/modules.sh 2>/dev/null || \
    echo "Warning: Modules system not initialized, but proceeding."

    # Load CUDA module
    module load cuda11.8 2>/dev/null || echo "Warning: Failed to load CUDA module. Proceeding with system defaults."
else
    echo "Warning: 'module' command not found. Proceeding with system defaults."
fi

# Check for CUDA compatibility
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA environment or GPU not detected. Exiting."
    exit 1
fi

# Print GPU information
nvidia-smi

# Run your Python script
python src/main.py --train

echo "Job completed successfully."
