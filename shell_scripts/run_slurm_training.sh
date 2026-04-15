#!/bin/bash -l
# FILENAME: run_slurm_training.sh

#SBATCH -A cis251356-ai
#SBATCH -p ai               # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=Resnet34_qr_training
#SBATCH --output=logs/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err    # separate error log (optional)

# Activate your Python environment
source .venv/bin/activate

cd /home/jhong90/github_proj/FlareTorch

python -u scripts/training.py
