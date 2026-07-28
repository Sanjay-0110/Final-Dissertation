#!/bin/bash
# scripts/slurm/train_job.sbatch
# ================================
# CSF3 GPU training jobscript (PROJECT_BRIEF.md §10).
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root, after env_setup.sh has been run once):
#     sbatch scripts/slurm/train_job.sbatch
# Monitor with:
#     squeue -u $USER
# Cancel with:
#     scancel <job_id>

#SBATCH --job-name=polyp-train
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8                    # matches training.num_workers in configs/config.yaml -- keep these in sync
#SBATCH --mem=32G
#SBATCH --time=16:00:00                      # rough estimate for 20 epochs; re-check after a 1-epoch test run (gpuA max is 4-00:00:00)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=gpu-free                   # GPU-specific account (vs. the general hum-msc-data-sci one)

set -euo pipefail

mkdir -p logs

# Environment modules are per-session -- compute nodes start with none
# loaded, so this must be repeated here even though env_setup.sh already
# loaded it once interactively. Use the SAME module name you confirmed via
# `module avail anaconda` and put in env_setup.sh.
module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line

# NOT `conda activate` -- only reliable on the login node interactively;
# jobscripts must use `source activate` (§10).
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

# Confirm configs/config.yaml's datasets.video.root still points at the
# correct path on CSF3 (e.g. ./datasets, relative to wherever you copied the
# repo) before this runs -- it was set for the local machine's layout.
python -m training.train --config configs/config.yaml
