#!/bin/bash
# Fine-tune Chronos on GPUS V100s with DDP (torchrun around Chronos' train.py; the Hugging Face Trainer inside shards
# the batches across the processes) and score checkpoint-final on the shared validation windows (CF-03).
#   CFG=chronos_ft_base_scaling GPUS=1 sbatch --gres=gpu:v100:1 slurm/chronos_ft_ddp.sbatch     (timing, 200 steps)
#   CFG=chronos_ft_base_scaling GPUS=2 sbatch --gres=gpu:v100:2 slurm/chronos_ft_ddp.sbatch
#   CFG=chronos_ft_base_5pct_nan GPUS=2 sbatch --gres=gpu:v100:2 slurm/chronos_ft_ddp.sbatch    (accuracy run)
# --gres on the command line must match GPUS. The config's per_device_train_batch_size is read as the effective batch
# (32) and passed to train.py as 32 / GPUS per GPU, so the effective batch and max_steps (optimizer steps) are the same
# on any GPU count. NAME (default <CFG>_g<GPUS>) names output/chronos/<NAME> (training), output/<NAME>.json (record) and
# results/<NAME>_val.json. EVAL=0 skips the scoring (timing runs). Compute nodes are offline: the model must be in the
# HF cache (load it once on a login node).
#SBATCH --job-name=chronos_ddp
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

CFG="${CFG:-chronos_ft_base_scaling}"
GPUS="${GPUS:-2}"
NAME="${NAME:-${CFG}_g${GPUS}}"
EVAL="${EVAL:-1}"
CHRONOS_REPO="${CHRONOS_REPO:-$HOME/scratch.msml612/chronos-forecasting}"
case "$CFG" in *_nan|*_scaling) CONTEXT=nan ;; *) CONTEXT=keep ;; esac
echo "$NAME ($CFG on $GPUS GPUs) on $(hostname), GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ';'), context $CONTEXT"
[ "$(nvidia-smi -L | wc -l)" -eq "$GPUS" ] || { echo "GPUS=$GPUS but $(nvidia-smi -L | wc -l) GPUs allocated"; exit 1; }
BATCH=$(grep -E "^per_device_train_batch_size:" "configs/${CFG}.yaml" | awk '{print $2}')
[ $((BATCH % GPUS)) -eq 0 ] || { echo "batch $BATCH does not split over $GPUS GPUs"; exit 1; }
echo "effective batch $BATCH, $((BATCH / GPUS)) per GPU"

T0=$SECONDS
torchrun --standalone --nproc_per_node "$GPUS" "$CHRONOS_REPO/scripts/training/train.py" \
    --config "configs/${CFG}.yaml" --output-dir "output/chronos/${NAME}" --per-device-train-batch-size $((BATCH / GPUS)) \
    || { echo "train.py failed"; exit 1; }
TRAIN_SECONDS=$((SECONDS - T0))
RUN=$(ls -d "output/chronos/${NAME}"/run-* | sort -V | tail -1)
echo "trained in ${TRAIN_SECONDS}s on $GPUS GPUs -> $RUN"
[ "$EVAL" = "1" ] || exit 0

python -m wpf.chronos_eval --model "$RUN/checkpoint-final" --name "$NAME" --context "$CONTEXT" \
    --out "output/${NAME}.json" --results results/ --train-seconds "$TRAIN_SECONDS" --train-config "configs/${CFG}.yaml"
