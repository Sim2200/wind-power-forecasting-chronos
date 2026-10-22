#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH -A msml612-class
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm-%j.out

source /etc/profile
module load python/3.10.10
unset PYTHONPATH
source "${ENV_DIR:-$HOME/envs/dl612env}/bin/activate"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

