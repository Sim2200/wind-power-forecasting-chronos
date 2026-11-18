#!/bin/bash
# Fine-tune Chronos on an SDWPF subset and score checkpoint-final on the shared validation windows.
#   CFG=chronos_ft_5pct sbatch slurm/chronos_ft.sbatch          (MIG slice, default)
#   CFG=chronos_ft_100pct sbatch --gpus=a100:1 slurm/chronos_ft.sbatch   (full A100 if the slice is too small)
#   CFG=chronos_ft_5pct_lr1e-5_nan SEED=2022 sbatch --gres=gpu:v100:1 slurm/chronos_ft.sbatch   (seed override)
# NAME (default <CFG>, or <CFG>_s<SEED> with a seed) names the training directory output/chronos/<NAME>, the record
# output/<NAME>.json with its predictions, and the shared-metric files results/<NAME>_<period>.json. PERIODS (default
# val) adds the one test pass for the grid: PERIODS="val test". Compute nodes are offline: the model must be in the HF cache
# (load it once on a login node).
#SBATCH --job-name=chronos_ft
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

CFG="${CFG:-chronos_ft_smoke}"
NAME="${NAME:-${CFG}${SEED:+_s$SEED}}"
PERIODS="${PERIODS:-val}"
CHRONOS_REPO="${CHRONOS_REPO:-$HOME/scratch.msml612/chronos-forecasting}"
case "$CFG" in *_nan) CONTEXT=nan ;; *) CONTEXT=keep ;; esac
echo "$NAME ($CFG${SEED:+, seed $SEED}) on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), context $CONTEXT, periods $PERIODS"

T0=$SECONDS
python "$CHRONOS_REPO/scripts/training/train.py" --config "configs/${CFG}.yaml" --output-dir "output/chronos/${NAME}" \
    ${SEED:+--seed $SEED} || { echo "train.py failed"; exit 1; }
TRAIN_SECONDS=$((SECONDS - T0))
RUN=$(ls -d "output/chronos/${NAME}"/run-* | sort -V | tail -1)
echo "trained in ${TRAIN_SECONDS}s -> $RUN"

python -m wpf.chronos_eval --model "$RUN/checkpoint-final" --name "$NAME" --context "$CONTEXT" --periods $PERIODS \
    --out "output/${NAME}.json" --results results/ --train-seconds "$TRAIN_SECONDS" --train-config "configs/${CFG}.yaml"
