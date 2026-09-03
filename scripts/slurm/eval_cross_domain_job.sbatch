#!/bin/bash
# scripts/slurm/eval_cross_domain_job.sbatch
# =============================================
# CSF3 GPU jobscript for cross-domain segmentation evaluation (inference
# only, no training) -- evaluation/evaluate_cross_domain.py.
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root):
#     sbatch scripts/slurm/eval_cross_domain_job.sbatch
# Monitor with:
#     squeue -u $USER
# Cancel with:
#     scancel <job_id>

#SBATCH --job-name=polyp-cross-eval
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00                       # inference only, no training -- should take a few minutes total (gpuA max is 4-00:00:00)
#SBATCH --output=logs/eval_cross_%j.out
#SBATCH --error=logs/eval_cross_%j.err
#SBATCH --account=gpu-free

set -euo pipefail

mkdir -p logs

module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

CHECKPOINT=checkpoints/proposed_segmentation_best.pt

echo "=== In-domain (Kvasir-SEG) ==="
python -m evaluation.evaluate_cross_domain --checkpoint "$CHECKPOINT" --dataset kvasir_seg

echo "=== Cross-domain (CVC-ClinicDB) ==="
python -m evaluation.evaluate_cross_domain --checkpoint "$CHECKPOINT" --dataset cvc_clinicdb
