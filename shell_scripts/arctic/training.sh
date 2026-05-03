#!/bin/bash
#SBATCH -p qGPU24
#SBATCH -A csc344r253
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=uqtraining
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Activate your Python environment
source /home/users/jhong36/miniforge3/etc/profile.d/conda.sh
cd /home/users/jhong36/projects/FlareTorch
conda activate flaretorch

python -u scripts/training.py
