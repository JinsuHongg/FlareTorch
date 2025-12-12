#!/bin/bash -l
# FILENAME: run_slurm.sh

#SBATCH -A cis251356-ai
#SBATCH -p ai               # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --job-name=Resnet34_MCD
#SBATCH --output=logs/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err    # separate error log (optional)


# Activate your Python environment
source .venv/bin/activate

# Move to your working directory
cd FlareTorch/src/FlareTorch/tasks

# Run your training script
python training.py --config-dir=/home/x-jhong6/project/FlareTorch/FlareTorch/configs --config-name="$1"
