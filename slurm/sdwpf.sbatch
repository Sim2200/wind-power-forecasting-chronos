#!/bin/bash
# PatchTST on SDWPF for one config in configs/. Submit from the repo root with the config name
# as an environment variable, e.g.:
#   CFG=sdwpf_smoke sbatch slurm/sdwpf.sbatch
#   CFG=sdwpf_5pct sbatch slurm/sdwpf.sbatch
#   CFG=sdwpf_5pct_lr3e-5 SEED=2022 sbatch slurm/sdwpf.sbatch      (seed override; the record is named <CFG>_s<SEED>)
# Results go to output/ (gitignored): <NAME>.json, <NAME>.pt and <NAME>_pred.npz, NAME defaulting to <CFG> or <CFG>_s<SEED>.
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
NAME="${NAME:-${CFG}${SEED:+_s$SEED}}"
echo "$NAME ($CFG${SEED:+, seed $SEED}) on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m wpf.train_sdwpf --config "configs/${CFG}.json" --out "output/${NAME}.json" ${SEED:+--seed $SEED}
