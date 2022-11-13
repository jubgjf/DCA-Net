#!/bin/bash

#SBATCH --mail-user=guanjiannan@outlook.com
#SBATCH --mail-type=END
#SBATCH -J DCA-Net                            # 作业名
#SBATCH -o slurm_logs/slurm-%j.out            # stdout 重定向，%j 会替换成 jobid
#SBATCH -e slurm_logs/slurm-%j.err            # stderr 重定向，%j 会替换成 jobid
#SBATCH -p compute                            # 作业提交的分区
#SBATCH -N 1                                  # 作业申请节点数量
#SBATCH -t 8:00:00                            # 任务运行的最长时间
#SBATCH --gres=gpu:tesla_p100-pcie-16gb:1

# source ~/.bashrc
source ~/.local/bin/miniconda3/etc/profile.d/conda.sh

# 设置运行环境
conda activate robust-slu

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python main_joint.py
