#!/bin/bash
# Fine-tune Chronos on an SDWPF subset and score checkpoint-final on the shared validation windows.
#   CFG=chronos_ft_5pct sbatch slurm/chronos_ft.sbatch          (MIG slice, default)
#   CFG=chronos_ft_100pct sbatch --gpus=a100:1 slurm/chronos_ft.sbatch   (full A100 if the slice is too small)
# The config's output_dir gets train.py's run-N folder; the record and predictions go to output/<CFG>.json / _pred.npz
# and the shared-metric files to results/<CFG>_val.json. Compute nodes are offline: the model must be in the HF cache
# (load it once on a login node).
#SBATCH --job-name=chronos_ft
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

CFG="${CFG:-chronos_ft_smoke}"
CHRONOS_REPO="${CHRONOS_REPO:-$HOME/scratch.msml612/chronos-forecasting}"
case "$CFG" in *_nan) CONTEXT=nan ;; *) CONTEXT=keep ;; esac
echo "$CFG on $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), context $CONTEXT"

T0=$SECONDS
python "$CHRONOS_REPO/scripts/training/train.py" --config "configs/${CFG}.yaml"
TRAIN_SECONDS=$((SECONDS - T0))
RUN=$(ls -d "output/chronos/${CFG}"/run-* | sort -V | tail -1)
echo "trained in ${TRAIN_SECONDS}s -> $RUN"

python -m wpf.chronos_eval --model "$RUN/checkpoint-final" --name "$CFG" --context "$CONTEXT" \
    --out "output/${CFG}.json" --results results/ --train-seconds "$TRAIN_SECONDS" --train-config "configs/${CFG}.yaml"
