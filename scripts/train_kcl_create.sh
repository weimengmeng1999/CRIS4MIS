#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/logs/%j.out
#SBATCH --job-name=cris4mis
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=262144
#SBATCH --time=2-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python train.py --config $CONFIG