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

# Use SLURM_SUBMIT_DIR to find the project root
# This is where you were when you ran 'sbatch'
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  # Fallback for local testing
  cd "$(dirname "$0")/../.."
fi

# Verify we are in the project root by checking for scripts/training.py
if [ ! -f "scripts/training.py" ]; then
  echo "Error: scripts/training.py not found in $(pwd)"
  echo "Please submit the job from the project root directory."
  exit 1
fi

# Environment activation
CONDA_SH="/home/users/jhong36/miniforge3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate flaretorch
elif command -v conda >/dev/null 2>&1; then
  conda activate flaretorch
else
  echo "Error: Conda not found at $CONDA_SH and 'conda' command not in PATH."
  exit 1
fi

# Verify environment and resources
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CWD: $(pwd)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
# We explicitly override trainer.devices and num_workers to match Slurm allocation
python -u scripts/training.py
