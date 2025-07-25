#!/bin/bash

# ==============================================================================
#                      SLURM Job Submission Script
# ==============================================================================
# This script is designed for efficient training on a SLURM cluster by:
# 1. Staging all necessary data (code, raw data, processed data) to a fast
#    node-local scratch disk (/tmp) to maximize I/O performance.
# 2. Running the training pipeline from the local disk.
# 3. Copying the final results (models, logs) back to permanent network storage.
# ==============================================================================

# --- SLURM DIRECTIVES ---
#SBATCH -J ChemulatorTrain              # Job name
#SBATCH -o ChemulatorTrain.o%j          # Standard output file
#SBATCH -e ChemulatorTrain.e%j          # Standard error file
#SBATCH -p gpu                          # Specify the GPU partition
#SBATCH --clusters=edge                 # Target the edge nodes
#SBATCH -N 1                            # Request a single node
#SBATCH -n 1                            # Run a single task
#SBATCH --gpus=1                        # Request one full GPU
#SBATCH --cpus-per-task=32              # Request 32 CPU cores
#SBATCH --mem=240G                      # Request 200 GB of CPU RAM
#SBATCH -t 24:00:00                     # Set a 24-hour runtime limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# --- SCRIPT CONFIGURATION ---
# Fail the job immediately if any command fails
set -e

# The root directory of your project on the permanent network filesystem.
# $SLURM_SUBMIT_DIR is the directory where you run 'sbatch' from.
PROJECT_ROOT_ON_NETWORK="$SLURM_SUBMIT_DIR"

# The name of your Conda environment.
CONDA_ENV_NAME="nn"

# ==============================================================================
#                             STEP 0: SETUP
# ==============================================================================
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname), Time: $(date)"
echo "========================================================"

# Create a unique temporary project root on the NODE-LOCAL scratch disk (/tmp).
# This is a high-speed, temporary storage space on the compute node itself.
LOCAL_PROJECT_ROOT=$(mktemp -d /tmp/chemulator_job_${SLURM_JOB_ID}.XXXXXX)
echo "Created temporary project root on local disk: $LOCAL_PROJECT_ROOT"

# Define a cleanup function to be called on any job exit signal (normal or error).
# This ensures the temporary directory is always removed.
cleanup() {
    echo "---
Cleaning up node-local scratch directory: $LOCAL_PROJECT_ROOT"
    rm -rf "$LOCAL_PROJECT_ROOT"
    echo "Cleanup complete."
}
trap cleanup EXIT

# ==============================================================================
#                      STEP 1: PREPARE ENVIRONMENT
# ==============================================================================
echo "---
--> Activating Conda environment..."
# Find and activate the specified conda environment.
CONDA_EXE=$(command -v conda) || { echo "Conda not found" >&2; exit 1; }
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
[ $? -ne 0 ] && { echo "Failed to activate conda env '$CONDA_ENV_NAME'" >&2; exit 1; }
echo "Conda environment '$CONDA_ENV_NAME' activated."

echo "---
--> Checking GPU and CUDA status..."
module load cuda11.8 || echo "Warning: 'module load cuda11.8' failed, using default."
nvidia-smi

# ==============================================================================
#                 STEP 2: STAGE PROJECT AND DATA TO LOCAL DISK
# ==============================================================================
echo "---
--> Staging project files to local disk for fast I/O..."

# Use rsync to robustly copy source code, config, and data directories.
# The '-a' flag (archive mode) is critical as it preserves permissions, ownership,
# and most importantly, file modification timestamps. Preserving timestamps
# prevents the Python script from recalculating the data hash unnecessarily.
# The '-vh --progress' flags provide verbose output and a progress bar.

echo "Copying src/ and config/ directories..."
rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/src/" "$LOCAL_PROJECT_ROOT/src/"
rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/config/" "$LOCAL_PROJECT_ROOT/config/"

# CRITICAL FIX: Copy both raw and processed data directories.
echo "Staging data directories (raw and processed)..."
if [ -d "$PROJECT_ROOT_ON_NETWORK/data/raw" ]; then
    mkdir -p "$LOCAL_PROJECT_ROOT/data/raw"
    rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/data/raw/" "$LOCAL_PROJECT_ROOT/data/raw/"
else
    echo "Warning: 'data/raw' directory not found on network."
fi

if [ -d "$PROJECT_ROOT_ON_NETWORK/data/processed" ]; then
    mkdir -p "$LOCAL_PROJECT_ROOT/data/processed"
    rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/data/processed/" "$LOCAL_PROJECT_ROOT/data/processed/"
else
    echo "Info: No existing 'data/processed' directory found. Preprocessing will run from scratch."
fi

echo "Staging complete."

# ==============================================================================
#                  STEP 3: RUN THE MAIN TRAINING SCRIPT
# ==============================================================================
# CRITICAL: Change to the local project root. All relative paths in the
# python script will now resolve to the fast local disk, not the slow network drive.
cd "$LOCAL_PROJECT_ROOT"

echo "---
--> Starting main pipeline from local directory: $(pwd)"

# Run the python script. Because we copied the processed data and preserved
# timestamps, the hashing mechanism will now detect the existing data and
# skip directly to the training phase (unless a config change requires it).
time python src/main.py --config config/config.jsonc

echo "Main training script finished."

# ==============================================================================
#          STEP 4: COPY FINAL RESULTS BACK TO PERMANENT NETWORK STORAGE
# ==============================================================================
echo "---
--> Copying final models and logs back to permanent network storage..."

# Use rsync to robustly copy the generated models and logs directories.
# We only need to sync these output directories back.
echo "Syncing data/models/ directory..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/data/models/" "$PROJECT_ROOT_ON_NETWORK/data/models/"

echo "Syncing logs/ directory..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/logs/" "$PROJECT_ROOT_ON_NETWORK/logs/"

# ==============================================================================
#                            JOB COMPLETION
# ==============================================================================
# The 'trap cleanup EXIT' command at the top will now execute automatically,
# removing the temporary directory from the compute node.
echo "========================================================"
echo "Job finished successfully: $(date)"
echo "========================================================"