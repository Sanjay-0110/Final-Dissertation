#!/bin/bash
# scripts/slurm/train_seg_ablation_job.sbatch
# ===============================================
# CSF3 GPU jobscript: §7a color-normalization ablation arm (normalization
# OFF). Otherwise identical setup to scripts/slurm/train_seg_job.sbatch
# (same warm-start encoder, same augmentation, same epoch/patience budget)
# so this isolates color_normalization as the only changed variable.
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root):
#     sbatch scripts/slurm/train_seg_ablation_job.sbatch
# Monitor with:
#     squeue -u $USER

#SBATCH --job-name=polyp-seg-ablation
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00                      # same budget as the normalization-on run this compares against (gpuA max is 4-00:00:00)
#SBATCH --output=logs/train_seg_ablation_%j.out
#SBATCH --error=logs/train_seg_ablation_%j.err
#SBATCH --account=gpu-free

set -euo pipefail

mkdir -p logs

module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

# --tag no_color_norm writes to checkpoints/proposed_segmentation_no_color_norm_{best,last}.pt
# -- does NOT overwrite the normalization-ON checkpoints from the main run.
python -m training.train_segmentation --config configs/config.yaml \
    --init-encoder-from checkpoints/proposed_classifier_best.pt \
    --disable-color-normalization \
    --tag no_color_norm
