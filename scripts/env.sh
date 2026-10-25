#!/bin/bash
# Shared environment setup for Zaratan. Source this from scripts and Slurm jobs:
#   source "$(dirname "$0")/env.sh"   (scripts)   /   source scripts/env.sh   (sbatch, after cd)
# The venv lives in home: scratch is fast for large files but very slow for the
# tens of thousands of small files a PyTorch install creates.
PY_MODULE="python/3.10.10"
ENV_DIR="${ENV_DIR:-$HOME/envs/dl612env}"

load_python() {
    source /etc/profile
    module load "$PY_MODULE"
    unset PYTHONPATH   # keep the module's preinstalled packages out of the venv
}

activate_env() {
    load_python
    source "$ENV_DIR/bin/activate"
}
