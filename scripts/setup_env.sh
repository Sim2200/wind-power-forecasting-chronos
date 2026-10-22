#!/bin/bash
# Create the project venv on Zaratan. Run once on a login node:
#   bash scripts/setup_env.sh
#
# The venv lives in home (~/envs): scratch is fast for large files but very slow
# for the tens of thousands of small files a PyTorch install creates.
set -eo pipefail   # no -u: /etc/profile and the module command use unset variables

ENV_DIR="${ENV_DIR:-$HOME/envs/dl612env}"
TORCH_VERSION="2.5.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

source /etc/profile
module load python/3.10.10
unset PYTHONPATH   # keep the module's preinstalled packages out of the venv

mkdir -p "$(dirname "$ENV_DIR")"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}" --extra-index-url "$TORCH_INDEX"
pip install -r "$(dirname "$0")/../requirements.txt"
echo "Environment ready at $ENV_DIR"
