#!/bin/bash
#SBATCH --partition=qGPU24
#SBATCH --account=csc344r253
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --job-name=uqcalibration
#SBATCH --output=calibration_%j.out
#SBATCH --error=calibration_%j.err

# Navigate to project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  cd "$SLURM_SUBMIT_DIR"
else
  cd "$(dirname "$0")/../.."
fi

# Verify script existence
if [ ! -f "scripts/calibration.py" ]; then
  echo "Error: scripts/calibration.py not found in $(pwd)"
  exit 1
fi

# Load environment
module load cuda/11.7
CONDA_SH="/home/users/jhong36/miniforge3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate flaretorch
else
  conda activate flaretorch
fi

# Run calibration
# Use hydra to override the config file name
python -u scripts/calibration.py --config-name QR_resnet18_calibration_surya_bench
