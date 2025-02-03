#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=220GB
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=aoberai_286

module purge
module load gcc/13.3.0
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate pytorch_env

tensorboard --logdir=/project/aoberai_286/RCD/FoundationModel_LiquidBiopsy/tensorboard_logs --host=0.0.0.0 --port=6006 &

python3 main.py