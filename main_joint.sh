#!/bin/bash

#SBATCH --mail-user=guanjiannan@outlook.com
#SBATCH --mail-type=END
#SBATCH -J DCA-Net
#SBATCH -o slurm_logs/slurm-%j.out
#SBATCH -e slurm_logs/slurm-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1

source ~/.local/bin/miniconda3/etc/profile.d/conda.sh
conda activate robust-slu

python main_joint.py --loss ce --user_mean 0 --user_std 1
python main_joint.py --loss ce
python main_joint.py --loss nce
python main_joint.py --loss sce
python main_joint.py --loss rce
python main_joint.py --loss nrce
python main_joint.py --loss gce
python main_joint.py --loss ngce
python main_joint.py --loss mae
python main_joint.py --loss nmae
python main_joint.py --loss nce rce
python main_joint.py --loss nce mae
python main_joint.py --loss gce nce
python main_joint.py --loss gce rce
python main_joint.py --loss gce mae
python main_joint.py --loss ngce nce
python main_joint.py --loss ngce rce
python main_joint.py --loss ngce mae
python main_joint.py --loss mae rce
