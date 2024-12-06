#!/bin/bash
#SBATCH --job-name="create-env"
#SBATCH --partition=compute-p1,compute-p2
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=research-ceg-he
#SBATCH --mem-per-cpu=8GB

# Load modules
module load 2023r1

# Ensure Miniforge is used
export PATH=$HOME/miniforge3/bin:$PATH
export CONDA_PKGS_DIRS="/scratch/${USER}/.conda/pkgs"
export CONDA_ENVS_DIRS="/scratch/${USER}/.conda/envs"

# Define environment variables
ENV_NAME="coastal-blue"
ENV_DIR="/scratch/${USER}/.conda/envs/$ENV_NAME"
YAML_FILE="$HOME/dev/coastpy/ci/envs/312-coastal-blue.yaml"

# Check if the YAML file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: Environment file $YAML_FILE does not exist."
    exit 1
fi

# Remove existing environment if it exists
if [ -d "$ENV_DIR" ]; then
    echo "Removing existing environment: $ENV_DIR"
    mamba env remove -p "$ENV_DIR" || echo "Environment $ENV_NAME does not exist, continuing..."
fi

# Create the new environment
echo "Creating environment $ENV_NAME from $YAML_FILE..."
mamba env create -f "$YAML_FILE" -p "$ENV_DIR" || {
    echo "Environment creation failed!"
    exit 1
}

# List environments to confirm
echo "Environment creation complete. Available environments:"
mamba env list

