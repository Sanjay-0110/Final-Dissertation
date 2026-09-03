#!/bin/bash
# scripts/slurm/eval_all_domains_job.sbatch
# ============================================
# CSF3 GPU jobscript: evaluates one checkpoint (inference only, no training)
# on all three datasets -- Kvasir-SEG (in-domain), CVC-ClinicDB and
# ETIS-LaribPolypDB (both cross-domain). Generic/reusable -- edit CHECKPOINT
# below to point at whichever checkpoint you want fully evaluated.
# CSF4/ALAN has NO GPUs -- this MUST run on CSF3, never CSF4.
#
# Submit with (from the repo root):
#     sbatch scripts/slurm/eval_all_domains_job.sbatch

#SBATCH --job-name=polyp-eval-all
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00                       # inference only across 3 datasets, ~1808 images total -- should take a few minutes (gpuA max is 4-00:00:00)
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --account=gpu-free

set -euo pipefail

mkdir -p logs

module load apps/binapps/anaconda3/2023.09   # TODO: match this to env_setup.sh's module line
source activate polyp-domain-robust

cd "$SLURM_SUBMIT_DIR"

# EDIT THIS before each submission to point at the checkpoint you want evaluated:
CHECKPOINT=checkpoints/proposed_segmentation_best.pt

echo "=== Checkpoint: $CHECKPOINT ==="
echo "=== In-domain: Kvasir-SEG ==="
python -m evaluation.evaluate_cross_domain --checkpoint "$CHECKPOINT" --dataset kvasir_seg

echo "=== Cross-domain: CVC-ClinicDB ==="
python -m evaluation.evaluate_cross_domain --checkpoint "$CHECKPOINT" --dataset cvc_clinicdb

echo "=== Cross-domain: ETIS-LaribPolypDB ==="
python -m evaluation.evaluate_cross_domain --checkpoint "$CHECKPOINT" --dataset etis_larib
