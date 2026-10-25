#!/bin/bash
# Create the project venv on Zaratan. Run once on a login node:
#   bash scripts/setup_env.sh
set -eo pipefail   # no -u: /etc/profile and the module command use unset variables

source "$(dirname "$0")/env.sh"
TORCH_VERSION="2.5.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

load_python
mkdir -p "$(dirname "$ENV_DIR")"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install "torch==${TORCH_VERSION}" --extra-index-url "$TORCH_INDEX"
pip install -r "$(dirname "$0")/../requirements.txt"
pip install -e "$(dirname "$0")/.."
echo "Environment ready at $ENV_DIR"
