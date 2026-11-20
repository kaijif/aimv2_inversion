#!/bin/bash
# One-time environment setup on a Longleaf LOGIN node (needs internet).
# Usage: bash setup_longleaf.sh
set -euo pipefail
WORK=/work/users/k/a/kaiji
ENV=$WORK/envs/fi
export PIP_CACHE_DIR=$WORK/pip_cache CONDA_PKGS_DIRS=$WORK/conda_pkgs

module load anaconda            # or a specific version, e.g. anaconda/2024.02 (see `module avail anaconda`)
eval "$(conda shell.bash hook)"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"

# CUDA 12.4 wheels work with any driver >= 525. Check `nvidia-smi` on a GPU node if in doubt.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.50,<5" huggingface_hub safetensors lpips torchmetrics \
            pandas matplotlib tqdm jupyter nbconvert ipykernel

# Makes the env selectable as a kernel in Open OnDemand Jupyter
python -m ipykernel install --user --name fi --display-name "Python (fi, aimv2 inversion)"

mkdir -p $WORK/aimv2_inversion/logs
python "$(dirname "$0")/prefetch_weights.py"
echo "done: activate with  module load anaconda && conda activate $ENV"
