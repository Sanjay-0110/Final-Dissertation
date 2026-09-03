#!/bin/bash
# scripts/slurm/train_seg_compression_asym_job.sbatch
# =======================================================
# CSF3 GPU jobscript: §7b ablation, ASYMMETRIC compression arm.
# From scratch (NOT warm-started) + augmentation -- companion to
# train_seg_compression_uniform_job.sbatch, which is identical except
# compression_mode=uniform. No warm-start here because the classifier
# checkpoint's encoder was trained under asymmetric channel widths and
# cannot be loaded into a uniform-mode encoder (shape mismatch) -- both
# arms run from scratch so compression_mode is the ONLY variable.
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root):
#     sbatch scripts/slurm/train_seg_compression_asym_job.sbatch

#SBATCH --job-name=polyp-seg-asym
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00                      # same budget as the epochs=100 warm-start run (gpuA max is 4-00:00:00)
#SBATCH --output=logs/train_seg_asym_%j.out
#SBATCH --error=logs/train_seg_asym_%j.err
#SBATCH --account=gpu-free

set -euo pipefail

mkdir -p logs

module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

# --tag asym_scratch_aug writes to checkpoints/proposed_segmentation_asym_scratch_aug_{best,last}.pt
python -m training.train_segmentation --config configs/config.yaml \
    --compression-mode asymmetric \
    --tag asym_scratch_aug
