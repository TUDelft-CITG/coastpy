#!/bin/bash
#SBATCH --job-name="coastal-grid"
#SBATCH --partition=compute-p1,compute-p2
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=research-ceg-he
#SBATCH --mem-per-cpu=3800M

# Load modules
module load 2023r1

# Ensure Miniforge is used
export PATH=$HOME/miniforge3/bin:$PATH
export CONDA_PKGS_DIRS="/scratch/${USER}/.conda/pkgs"
export CONDA_ENVS_DIRS="/scratch/${USER}/.conda/envs"

# Define environment variables
ENV_NAME="coastal-blue"
ENV_DIR="/scratch/${USER}/.conda/envs/$ENV_NAME"

echo "Initializing Miniforge environment..."
source $HOME/miniforge3/etc/profile.d/conda.sh || {
    echo "Failed to initialize Conda"
    exit 1
}

echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME" || {
    echo "Failed to activate Conda environment: $ENV_NAME"
    exit 1
}

# Execute the Python script
SCRIPT="$HOME/dev/coastpy/scripts/python/make_coastal_grid.py"
echo "Running Python script: $SCRIPT"
python "$SCRIPT" --zoom 5 6 7 8 9 10 --buffer_size 500m 1000m 2000m 5000m 10000m 15000m --release 2025-01-01 --verbose || {
    echo "Python script execution failed"
    exit 1
}

echo "Processing completed successfully."
