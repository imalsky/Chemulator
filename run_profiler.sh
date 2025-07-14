#!/bin/bash
#SBATCH -J MYGPU-profile-5min # Job name: Clear and descriptive
#SBATCH -o MYGPUJOB.o%j     # Name of job output file
#SBATCH -e MYGPUJOB.e%j     # Name of stderr error file
#SBATCH -p gpu              # Queue (partition) name for GPU nodes
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH --clusters=edge     # Directs the job to the edge cluster nodes
#SBATCH --cpus-per-gpu=32   # Number of CPU cores
#SBATCH --gpus=1            # Request 1 GPU
#SBATCH --mem=79G           # Memory (RAM) requested
#SBATCH -t 00:10:00         # MODIFIED: 10-minute wall time provides a safe buffer
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

echo "--- Job Environment Setup ---"
echo "Job starting on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"

# Change to the directory from which the script was submitted
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# --- Load Required Modules ---
echo "Loading modules..."
if command -v module &> /dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || echo "Warning: Could not source modules.sh"
    module purge
    # Load the HPC SDK which contains the Nsight Systems profiler (nsys)
    module load nvhpc/24.9
    echo "Loaded nvhpc/24.9"
else
    echo "Fatal: 'module' command not found. Cannot load profiler. Exiting."
    exit 1
fi

# --- Activate Conda Environment ---
echo "Initializing and activating Conda environment 'nn'..."
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting."
    exit 1
fi
CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'. Exiting."; exit 1; }
echo "Python executable: $(which python)"

# --- Pre-run Sanity Check ---
echo "--- GPU Information ---"
nvidia-smi

# --- Run Profiling Command ---
echo "--- Starting Profiling with Nsight Systems ---"
echo "Profiling will start after a 60-second delay and run for 180 seconds."

# MODIFIED: The 'nsys profile' command now controls the runtime.
# --delay=60: Wait 60s for the script to initialize (data loading, torch.compile) before profiling.
# --duration=180: Profile for 180s (3 minutes), then gracefully stop the script and write the report.
# --trace=...: Captures GPU, NVTX (PyTorch markers), and OS runtime (file IO) activity.
# -o "report_%j": Creates a unique report file named with the job ID.

nsys profile --trace=cuda,nvtx,osrt --delay=60 --duration=180 -w true -o "report_${SLURM_JOB_ID}" --force-overwrite=true \
python src/main.py

echo "--- Job Completion ---"
echo "Nsight Systems has completed profiling."
echo "Report file generated: report_${SLURM_JOB_ID}.nsys-rep"
echo "Job completed successfully."
