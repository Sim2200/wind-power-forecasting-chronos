#!/bin/bash
# PatchTST on SDWPF for one config in configs/. Submit from the repo root with the config name
# as an environment variable, e.g.:
#   CFG=sdwpf_smoke sbatch slurm/sdwpf.sbatch
#   CFG=sdwpf_5pct sbatch slurm/sdwpf.sbatch
# Results go to output/ (gitignored): <CFG>.json, <CFG>.pt and <CFG>_pred.npz.
#SBATCH --job-name=sdwpf
#SBATCH -A msml612-class
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --output=slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CFG="${CFG:-sdwpf_smoke}"
echo "$CFG on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m wpf.train_sdwpf --config "configs/${CFG}.json" --out "output/${CFG}.json"
