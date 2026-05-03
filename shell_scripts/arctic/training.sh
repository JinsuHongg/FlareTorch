#!/bin/bash
#SBATCH -p qGPU24
#SBATCH -A csc344r253
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=uqtraining
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Activate your Python environment
# Try to find conda in common locations
CONDA_PROFILE=$(find /home/users/$USER -name conda.sh -path "*/etc/profile.d/conda.sh" 2>/dev/null | head -n 1)
if [ -z "$CONDA_PROFILE" ]; then
  # Fallback to a common path if find fails
  CONDA_PROFILE="$HOME/miniforge3/etc/profile.d/conda.sh"
fi

if [ -f "$CONDA_PROFILE" ]; then
  source "$CONDA_PROFILE"
  conda activate flaretorch
else
  echo "Error: Could not find conda.sh"
  exit 1
fi

# Move to the project root directory
cd ${SLURM_SUBMIT_DIR:-.}

# Run training with Hydra overrides to match Slurm allocation
python -u scripts/training.py
