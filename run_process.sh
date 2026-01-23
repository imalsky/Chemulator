#!/bin/bash
#SBATCH -J preprocess
#SBATCH -o preprocess.o%j
#SBATCH -e preprocess.e%j
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

cd "$SLURM_SUBMIT_DIR"

# Conda setup
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Exiting."
    exit 1
fi
CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate nn || { echo "Error: Failed to activate Conda environment 'nn'. Exiting."; exit 1; }

# Stability settings for HDF5
export HDF5_USE_FILE_LOCKING=FALSE

# Run preprocessing
echo "Starting preprocessing..."
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

python -u processing/preprocessing.py

echo "Preprocessing complete."