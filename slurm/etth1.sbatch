#!/bin/bash
# Full PatchTST run on ETTh1 for one config in configs/. Submit from the repo root with
# the config name as an environment variable, e.g.:
#   CFG=etth1_h192 sbatch slurm/etth1.sbatch
#   CFG=etth1_h96_mix sbatch slurm/etth1.sbatch   (ablation)
#SBATCH --job-name=etth1
#SBATCH -A msml612-class
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CFG="${CFG:-etth1_h96}"
echo "$CFG on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m wpf.train --config "configs/${CFG}.json" --out "results/${CFG}.json"

