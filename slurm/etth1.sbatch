#!/bin/bash
# Full PatchTST run on ETTh1 for one forecast length. Submit from the repo root with
# the horizon as an environment variable, e.g.:  H=96 sbatch slurm/etth1.sbatch
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

H="${H:-96}"
echo "ETTh1 H=$H on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m wpf.train --config "configs/etth1_h${H}.json" --out "results/etth1_h${H}.json"

