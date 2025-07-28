#!/bin/bash

# ==============================================================================
#                 SLURM Job Submission Script - Chemulator Pipeline
# ==============================================================================
# This script ensures data normalization is completed and saved before training.
# 
# Usage:
#   sbatch run.sh [train|tune] [--trials N]
#   
# 
# Examples:
#   sbatch run.sh train              # Normalize then train
#   sbatch run.sh tune --trials 100  # Normalize then tune with 100 trials
# ==============================================================================

# --- SLURM DIRECTIVES ---
#SBATCH -J ChemulatorJob                # Job name
#SBATCH -o ChemulatorJob.o%j            # Standard output file
#SBATCH -e ChemulatorJob.e%j            # Standard error file
#SBATCH -p gpu                          # Specify the GPU partition
#SBATCH --clusters=edge                 # Target the edge nodes
#SBATCH -N 1                            # Request a single node
#SBATCH -n 1                            # Run a single task
#SBATCH --gpus=1                        # Request one full GPU
#SBATCH --cpus-per-task=32              # Request 32 CPU cores
#SBATCH --mem=240G                      # Request 240 GB of CPU RAM
#SBATCH -t 72:00:00                     # Set a 48-hour runtime limit (for tuning)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# --- SCRIPT CONFIGURATION ---
set -e  # Exit on error

# Parse command line arguments
MODE="${1:-train}"  # Default to train if not specified
TRIALS=200          # Default trials for tuning

# Parse additional arguments
shift || true  # Shift past the mode argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ "$MODE" != "train" && "$MODE" != "tune" ]]; then
    echo "Error: MODE must be 'train' or 'tune', got: $MODE"
    exit 1
fi

PROJECT_ROOT_ON_NETWORK="$SLURM_SUBMIT_DIR"
CONDA_ENV_NAME="nn"
STUDY_NAME="chemulator_hpo_${SLURM_JOB_ID}"

# ==============================================================================
#                             STEP 0: SETUP
# ==============================================================================
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname), Time: $(date)"
echo "Mode: $MODE"
[[ "$MODE" == "tune" ]] && echo "Trials: $TRIALS"
echo "========================================================"

# Create local working directory
LOCAL_PROJECT_ROOT=$(mktemp -d /tmp/chemulator_job_${SLURM_JOB_ID}.XXXXXX)
echo "Created temporary project root on local disk: $LOCAL_PROJECT_ROOT"

# Cleanup function
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
CONDA_EXE=$(command -v conda) || { echo "Conda not found" >&2; exit 1; }
CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
[ $? -ne 0 ] && { echo "Failed to activate conda env '$CONDA_ENV_NAME'" >&2; exit 1; }
echo "Conda environment '$CONDA_ENV_NAME' activated."

echo "---
--> Checking GPU and CUDA status..."
# Try to load CUDA module if available
if command -v module &> /dev/null; then
    module load cuda11.8 2>/dev/null || echo "Warning: 'module load cuda11.8' failed, using default."
fi
nvidia-smi

# ==============================================================================
#                    STEP 2: STAGE PROJECT FILES TO LOCAL DISK
# ==============================================================================
echo "---
--> Staging project files to local disk for fast I/O..."

# Copy source code and config
echo "Copying src/ and config/ directories..."
rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/src/" "$LOCAL_PROJECT_ROOT/src/"
rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/config/" "$LOCAL_PROJECT_ROOT/config/"

# Create data directories
mkdir -p "$LOCAL_PROJECT_ROOT/data/raw"
mkdir -p "$LOCAL_PROJECT_ROOT/data/processed"
mkdir -p "$LOCAL_PROJECT_ROOT/data/models"
mkdir -p "$LOCAL_PROJECT_ROOT/logs"

# Copy raw data files (required for normalization)
echo "Copying raw data files..."
if [ -d "$PROJECT_ROOT_ON_NETWORK/data/raw" ]; then
    rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/data/raw/" "$LOCAL_PROJECT_ROOT/data/raw/"
else
    echo "ERROR: No raw data found at $PROJECT_ROOT_ON_NETWORK/data/raw"
    exit 1
fi

# Copy existing processed data if it exists (might allow skipping normalization)
echo "Checking for existing processed data..."
if [ -d "$PROJECT_ROOT_ON_NETWORK/data/processed" ]; then
    echo "Found existing processed data, copying..."
    rsync -avh --progress "$PROJECT_ROOT_ON_NETWORK/data/processed/" "$LOCAL_PROJECT_ROOT/data/processed/"
else
    echo "No existing processed data found. Will create during normalization."
fi

echo "Staging complete."

# ==============================================================================
#                    STEP 3: RUN DATA NORMALIZATION
# ==============================================================================
cd "$LOCAL_PROJECT_ROOT"

echo "---
--> Running data normalization step..."
echo "This ensures processed data exists and is saved before any training begins."

# Run normalization (this will check hash and skip if data is current)
time python src/main.py --config config/config.jsonc --normalize

# Immediately sync normalized data back to network storage
echo "---
--> Syncing normalized data back to network storage..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/data/processed/" "$PROJECT_ROOT_ON_NETWORK/data/processed/"
echo "Normalized data safely stored on network."

# ==============================================================================
#                    STEP 4: RUN TRAINING OR TUNING
# ==============================================================================
echo "---
--> Starting $MODE phase..."

if [[ "$MODE" == "train" ]]; then
    echo "Running standard training..."
    time python src/main.py --config config/config.jsonc --train
elif [[ "$MODE" == "tune" ]]; then
    echo "Running hyperparameter optimization with $TRIALS trials..."
    time python src/main.py --config config/config.jsonc --tune --trials "$TRIALS" --study-name "$STUDY_NAME"
    
    # Copy Optuna database back
    if [ -f "${STUDY_NAME}.db" ]; then
        echo "Copying Optuna study database back to network..."
        cp "${STUDY_NAME}.db" "$PROJECT_ROOT_ON_NETWORK/"
    fi
    
    # Copy Optuna results
    if [ -d "optuna_results" ]; then
        echo "Copying Optuna results back to network..."
        rsync -avh --progress "optuna_results/" "$PROJECT_ROOT_ON_NETWORK/optuna_results/"
    fi
fi

echo "$MODE phase completed."

# ==============================================================================
#          STEP 5: COPY FINAL RESULTS BACK TO PERMANENT NETWORK STORAGE
# ==============================================================================
echo "---
--> Copying final results back to permanent network storage..."

# Always sync models
echo "Syncing data/models/ directory..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/data/models/" "$PROJECT_ROOT_ON_NETWORK/data/models/"

# Always sync logs
echo "Syncing logs/ directory..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/logs/" "$PROJECT_ROOT_ON_NETWORK/logs/"

# Sync any updates to processed data (in case of new splits, etc.)
echo "Final sync of data/processed/ directory..."
rsync -avh --progress "$LOCAL_PROJECT_ROOT/data/processed/" "$PROJECT_ROOT_ON_NETWORK/data/processed/"

# ==============================================================================
#                            JOB COMPLETION
# ==============================================================================
echo "========================================================"
echo "Job finished successfully: $(date)"
echo "Results saved to: $PROJECT_ROOT_ON_NETWORK"
echo "========================================================"