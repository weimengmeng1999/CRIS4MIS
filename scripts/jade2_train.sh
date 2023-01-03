#!/bin/bash -l
#SBATCH --output=/jmain02/home/J2AD019/exk01/%u/logs/%j.out
#SBATCH --job-name=cris
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00

source ~/.bashrc
module load cuda
nvidia-smi -i $CUDA_VISIBLE_DEVICES
nvcc --version

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python train.py --config $CONFIG