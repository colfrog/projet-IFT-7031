#!/bin/bash
#SBATCH --account=def-csubakan-ab
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-9:59            # Runtime in D-HH:MM
#SBATCH --mem=64000               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/home/lcimon/scratch/slurm_logs/%j.out  # File to which STDOUT will be written
#SBATCH --error=/home/lcimon/scratch/slurm_logs/%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU

# sbatch run.sh

scontrol show job=$SLURM_JOB_ID
source ~/.bashrc
source .venv/bin/activate
export HF_HOME=/home/lcimon/scratch
export KAGGLEHUB_CACHE=/home/lcimon/scratch/kaggle

python qwen2-finetuning.py

