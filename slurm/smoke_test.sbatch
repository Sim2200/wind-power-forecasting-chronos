#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH -A msml612-class
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

