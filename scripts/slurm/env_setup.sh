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

# TODO: confirm the exact anaconda module name available on CSF3 -- run
#     module avail anaconda
# on the login node and replace the line below with the real module path.
module load apps/binapps/anaconda3/2023.09

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
source activate "$ENV_NAME"

# TODO: confirm the CUDA version on CSF3's GPU nodes before relying on
# requirements.txt's torch pin (torch==2.13.0) resolving correctly here --
# run `module avail cuda` on the login node, or check `nvidia-smi` on a GPU
# node once allocated. If the pinned version doesn't resolve, use the
# CUDA-matched install command from pytorch.org's official selector instead
# of a plain `pip install torch`.
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment '$ENV_NAME' created."
echo "Activate it in jobscripts with (NOT 'conda activate' -- see §10):"
echo "    source activate $ENV_NAME"
