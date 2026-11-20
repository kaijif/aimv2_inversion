#!/bin/bash
# CPU-only prep: list ImageNet, draw splits, decode all images into the pixel cache.
# Runs on the general partition (no GPU queue). The GPU job then finds the cache and starts training.
#   sbatch prep_longleaf.sbatch                                   # full run (50k train)
#   sbatch --export=ALL,FI_TRAIN_LIMIT=10000 prep_longleaf.sbatch # milestone-1 cache
#SBATCH --job-name=aimv2-prep
#SBATCH -p general
#SBATCH -n 1
#SBATCH --cpus-per-task=17
#SBATCH --mem=32g
#SBATCH -t 04:00:00
#SBATCH -o /work/users/k/a/kaiji/aimv2_inversion/logs/%x-%j.out
set -euo pipefail

WORK=/work/users/k/a/kaiji
PROJ=$WORK/aimv2_inversion
module load anaconda
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fi

export HF_HOME=$WORK/hf_cache TORCH_HOME=$WORK/torch_cache MPLCONFIGDIR=$WORK/mpl_cache
export PYTHONUNBUFFERED=1

cd "$PROJ"
python prep_data.py aimv2_feature_inversion.ipynb
