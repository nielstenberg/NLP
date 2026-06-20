#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:59
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name=generating-900-traces

module purge

export PYTHONUNBUFFERED=1

source /scratch/s5208475/NLP/trace_generation/bin/activate
export HF_TOKEN="your_token_here"

python3 model.py