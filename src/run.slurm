#!/bin/bash
#SBATCH -J cpy
#SBATCH -p xhhgnormal
#SBATCH -N 1
#SBATCH -n 16
##SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1   #根据需要调整卡数,满足核卡比=16:1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rdkit_torch1131py39
module load  nvidia/cuda/11.6
python graph_prop_pred.py #需要运行的程序
