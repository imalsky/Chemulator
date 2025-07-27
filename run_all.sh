#!/bin/bash
#SBATCH -J MYGPUJOB         # Job name
#SBATCH -o MYGPUJOB.o%j     # Name of job output file
#SBATCH -e MYGPUJOB.e%j     # Name of stderr error file
#SBATCH -p gpu              # Queue (partition) name for GPU nodes
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH -n 4                # Total # of CPU cores (adjust as needed)
#SBATCH --clusters=edge     # *** CRITICAL: Directs the job to the edge cluster nodes (gn11-gn14) ***
#SBATCH --cpus-per-gpu=24
#SBATCH --gpus=1            # Request 1 GPU (adjust if more needed)
#SBATCH --mem=79G           # Memory (RAM) requested for gpu-mig
#SBATCH -t 24:00:00         # Run time (hh:mm:ss) for gpu-mig (adjust if needed)
#SBATCH --mail-type=all     # Send email at begin and end of job
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
python src/main.py

echo "Job completed successfully."