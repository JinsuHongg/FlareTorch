#!/bin/bash -l
# FILENAME: run_slurm_pixelsum_testset.sh

##SBATCH -A cis251356-ai
#SBATCH -p standard               # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
##SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --job-name=img_pixel_sum_testset
#SBATCH --output=logs/%x_%j.out   # %x = job name, %j = job ID
#SBATCH --error=logs/%x_%j.err    # separate error log (optional)


# Activate your Python environment
source .venv/bin/activate

# Move to your working directory
cd FlareTorch/src/FlareTorch/utils

# Run your training script
python -u compute_sum_pixels_for_each_data.py --config-dir=/home/x-jhong6/project/FlareTorch/FlareTorch/configs --config-name="$1"
