#!/bin/bash
# LoRA fine-tune Chronos on an SDWPF subset (scripts/train_lora.py around Chronos' train.py) and score the merged
# checkpoint-final on the shared validation windows.
#   CFG=chronos_lora_5pct_lr1e-4_nan RANK=8 sbatch slurm/chronos_lora.sbatch                 (MIG slice, default)
#   CFG=chronos_lora_5pct_lr1e-4_nan RANK=8 sbatch --gpus=v100:1 --gres=gpu:v100:1 slurm/chronos_lora.sbatch
# NAME (default <CFG>_r<RANK>) names the training directory output/chronos/<NAME>, the record output/<NAME>.json, the
# predictions and results/<NAME>_val.json, so the rank sweep (#59) reuses one config and concurrent jobs never share a
# run-N directory. Compute nodes are offline: the model must be in the HF cache.
#SBATCH --job-name=chronos_lora
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
CFG="${CFG:-chronos_lora_smoke}"
RANK="${RANK:-8}"
NAME="${NAME:-${CFG}_r${RANK}}"
export CHRONOS_REPO="${CHRONOS_REPO:-$HOME/scratch.msml612/chronos-forecasting}"
case "$CFG" in *_nan|*_smoke) CONTEXT=nan ;; *) CONTEXT=keep ;; esac
echo "$NAME ($CFG, rank $RANK) on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), context $CONTEXT"

T0=$SECONDS
python scripts/train_lora.py --config "configs/${CFG}.yaml" --rank "$RANK" --output-dir "output/chronos/${NAME}" \
    || { echo "train_lora.py failed"; exit 1; }
TRAIN_SECONDS=$((SECONDS - T0))
RUN=$(ls -d "output/chronos/${NAME}"/run-* | sort -V | tail -1)
echo "trained in ${TRAIN_SECONDS}s -> $RUN"

python -m wpf.chronos_eval --model "$RUN/checkpoint-final" --name "$NAME" --context "$CONTEXT" \
    --out "output/${NAME}.json" --results results/ --train-seconds "$TRAIN_SECONDS" --train-config "configs/${CFG}.yaml" \
    --train-info "$RUN/checkpoint-final/lora.json"
