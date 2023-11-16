#!/bin/zsh

#SBATCH -J DCA-Net
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:geforce_rtx_2080_ti:1
#SBATCH -w gpu17

source ~/.miniconda/etc/profile.d/conda.sh
conda activate hpc

wandb agent jubgjf/robust-slu-small/9noxdd7x
