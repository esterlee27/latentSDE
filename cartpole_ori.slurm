#!/bin/bash
#SBATCH --job-name=original              # Name of your job
#SBATCH --nodes=1
#SBATCH --output=slurm-%j.out             # Standard output file (%j = job ID)
#SBATCH --error=slurm-%j.err              # Standard error file
#SBATCH --partition=swarm_h100                  # Use the A100 GPU partition (change if needed)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --time=1-12:00:00                   # Max runtime (1 hour)
#SBATCH --mem=8G                          # Memory allocation
#SBATCH --cpus-per-task=4                 # Number of CPU cores allocated to your task

# Load Conda module
module load conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate Sto_BC

# Navigate to your working directory (current directory is /home/yl1d24)
cd /iridisfs/scratch/yl1d24/

# Run your Python script (replace with your actual script name)
python cartpole_ori.py
