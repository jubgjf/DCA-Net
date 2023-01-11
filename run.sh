#!/bin/zsh

#SBATCH --mail-user=guanjiannan@outlook.com
#SBATCH --mail-type=END
#SBATCH -J DCA-Net
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1

source ~/.local/bin/miniconda3/etc/profile.d/conda.sh
conda activate robust-slu

WANDB_MODE="offline"
python run.py
