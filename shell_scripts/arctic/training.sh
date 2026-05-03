#!/bin/bash
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=uqtraining
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Determine project root relative to script location
# Assumes script is at shell_scripts/arctic/training.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Environment activation
# Using the path from the original script as the primary source
CONDA_SH="/home/users/jhong36/miniforge3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  # Try finding conda in the path if sourcing failed
  if command -v conda >/dev/null 2>&1; then
    echo "Using system conda"
  else
    echo "Error: Conda not found. Please check your conda installation path."
    exit 1
  fi
fi

conda activate flaretorch

# Verify environment and resources
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CWD: $(pwd)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
# We explicitly override trainer.devices and num_workers to match Slurm allocation
python -u scripts/training.py
