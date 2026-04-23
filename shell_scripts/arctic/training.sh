#!/bin/bash
#SBATCH -p qCPU24
#SBATCH -A csc344r253
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=my_job
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Activate your Python environment
source .venv/bin/activate

cd /home/jhong90/github_proj/FlareTorch

python -u scripts/training.py
