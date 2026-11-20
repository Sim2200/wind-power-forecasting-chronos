#!/bin/bash
# EX-01: one array task = one cell of the experiment grid (wpf.grid). Submit with
#   python -m wpf.grid submit            (two arrays: the MIG cells and the V100 cells; --dry-run prints them)
# GRID_PARTITION (mig | v100) and the array index pick the cell; its env (CFG, SEED, NAME, PERIODS, ...) is evaluated
# and the model's own sbatch script is run inside this allocation, so every model keeps its single-run entry point.
# The GPU is given on the submit command line (wpf.grid.PARTITIONS); the header's MIG slice is the fallback.
#SBATCH --job-name=grid
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-grid-%A_%a.out
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
activate_env
: "${GRID_PARTITION:?set GRID_PARTITION=mig|v100 (python -m wpf.grid submit does)}"
eval "$(python -m wpf.grid cell --partition "$GRID_PARTITION" --index "$SLURM_ARRAY_TASK_ID")"
echo "grid cell $GRID_PARTITION[$SLURM_ARRAY_TASK_ID] = $NAME via $SCRIPT"
bash "$SCRIPT"
