#!/bin/bash
# scripts/slurm/env_setup.sh
# ============================
# One-time conda environment setup on CSF3's login node (PROJECT_BRIEF.md §10).
# Run this ONCE, interactively, on the CSF3 login node -- do NOT submit this
# as a SLURM job (it just installs packages; no compute needed).
#
# Installs into a self-contained conda env under $HOME -- never modifies the
# central Anaconda install.
#
# Usage (on the CSF3 login node, from the repo root):
#     bash scripts/slurm/env_setup.sh

set -euo pipefail

ENV_NAME="polyp-domain-robust"
PYTHON_VERSION="3.11"

# Module names change over time -- if this one is gone, find the current one
# with `module avail anaconda` and update it here and in every .sbatch file.
module load apps/binapps/anaconda3/2023.09

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
source activate "$ENV_NAME"

# If requirements.txt's torch pin doesn't resolve for the GPU nodes' CUDA
# version (check `nvidia-smi` on a GPU node), use the
# CUDA-matched install command from pytorch.org's official selector instead
# of a plain `pip install torch`.
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment '$ENV_NAME' created."
echo "Activate it in jobscripts with (NOT 'conda activate' -- see §10):"
echo "    source activate $ENV_NAME"
