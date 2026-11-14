#!/bin/bash
# Score a Chronos checkpoint on the shared validation windows without training (zero-shot, or a saved checkpoint).
#   MODEL=amazon/chronos-t5-small NAME=chronos_small_zeroshot sbatch slurm/chronos_eval.sbatch
#   CONTEXT=nan and PERIODS="val test" override the defaults. The model must already be in the HF cache.
#SBATCH --job-name=chronos_eval
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

MODEL="${MODEL:-amazon/chronos-t5-small}"
NAME="${NAME:-chronos_small_zeroshot}"
CONTEXT="${CONTEXT:-keep}"
PERIODS="${PERIODS:-val}"
echo "$NAME: $MODEL on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader), context $CONTEXT, periods $PERIODS"
python -m wpf.chronos_eval --model "$MODEL" --name "$NAME" --context "$CONTEXT" --periods $PERIODS \
    --out "output/${NAME}.json" --results results/
