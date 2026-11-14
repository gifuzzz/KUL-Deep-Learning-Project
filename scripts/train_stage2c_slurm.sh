#!/usr/bin/env bash
#SBATCH --job-name=rescue_stage2c
#SBATCH --output=logs/stage2c_%j.out
#SBATCH --error=logs/stage2c_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ==============================================================================
# FlameBox Stage 2C (Retention-Aware Distance Bridge) Slurm Submission Script
# ==============================================================================
set -euo pipefail

mkdir -p logs models/curriculum_stage2c_production

echo "=============================================================================="
echo "Starting Stage 2C Production Training on Slurm"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : ${SLURM_NODENAME:-$(hostname)}"
echo "Start  : $(date)"
echo "=============================================================================="

# Optional cluster module loads (uncomment and adapt if your cluster requires)
# module purge || true
# module load cuda/12.1 python/3.11 || true

bash scripts/train_hpc.sh \
    --stage 2C \
    --stage2c-mode retention_aware \
    --stage2c-timesteps 500000 \
    --architecture baseline \
    --checkpoint models/curriculum_stage2/stage2/stage2_final.zip \
    --save-dir models/curriculum_stage2c_production \
    --seed 42 \
    --n-envs 8 \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    "$@"

echo "=============================================================================="
echo "End    : $(date)"
echo "=============================================================================="
