#!/bin/bash
# scripts/slurm/train_seg_compression_uniform_job.sbatch
# ==========================================================
# CSF3 GPU jobscript: §7b ablation, UNIFORM compression arm.
# From scratch + augmentation -- identical setup to
# train_seg_compression_asym_job.sbatch except compression_mode=uniform, so
# that flag is the only variable between the two runs.
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root):
#     sbatch scripts/slurm/train_seg_compression_uniform_job.sbatch

#SBATCH --job-name=polyp-seg-uniform
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00                      # same budget as the asymmetric arm (gpuA max is 4-00:00:00)
#SBATCH --output=logs/train_seg_uniform_%j.out
#SBATCH --error=logs/train_seg_uniform_%j.err
#SBATCH --account=gpu-free

set -euo pipefail

mkdir -p logs

module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

# --tag uniform_scratch_aug writes to checkpoints/proposed_segmentation_uniform_scratch_aug_{best,last}.pt
python -m training.train_segmentation --config configs/config.yaml \
    --compression-mode uniform \
    --tag uniform_scratch_aug
