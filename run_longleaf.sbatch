#!/bin/bash
# Headless run of the notebook on a Longleaf GPU node.
#   mkdir -p /work/users/k/a/kaiji/aimv2_inversion/logs      (once; Slurm needs the log dir to exist)
#   cp aimv2_feature_inversion.ipynb run_notebook.py /work/users/k/a/kaiji/aimv2_inversion/
#   sbatch run_longleaf.sbatch                                                  # full run
#   sbatch --export=ALL,FI_TRAIN_LIMIT=10000,FI_ENCODERS=aimv2 run_longleaf.sbatch   # milestone 1
#   sbatch --export=ALL,FI_SMOKE=1 -t 01:00:00 run_longleaf.sbatch              # plumbing test
# Every stage resumes (training per epoch), so a timed-out job can just be resubmitted, or chained:
#   j=$(sbatch --parsable run_longleaf.sbatch); sbatch --dependency=afterany:$j run_longleaf.sbatch
#SBATCH --job-name=aimv2-inv
#SBATCH -p a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 2-00:00:00
#SBATCH -o /work/users/k/a/kaiji/aimv2_inversion/logs/%x-%j.out
set -euo pipefail

WORK=/work/users/k/a/kaiji
PROJ=$WORK/aimv2_inversion
module load anaconda
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fi

export HF_HOME=$WORK/hf_cache TORCH_HOME=$WORK/torch_cache MPLCONFIGDIR=$WORK/mpl_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # weights were prefetched on a login node
export PYTHONUNBUFFERED=1 FI_PROGRESS=0   # no progress bars in the log

cd "$PROJ"
nvidia-smi
python run_notebook.py aimv2_feature_inversion.ipynb
