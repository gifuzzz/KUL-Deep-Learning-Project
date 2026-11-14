#!/usr/bin/env bash
#SBATCH --job-name=rescue_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

# ==============================================================================
# FlameBox Architecture Ablation Slurm Job Array Script
# Runs 3 architectures in parallel across cluster nodes:
#   Index 0: baseline (128-unit)
#   Index 1: deep128  (Deeper fusion, 128-unit GRU)
#   Index 2: deep256  (Expanded fusion, 256-unit GRU, MLP heads)
# ==============================================================================
set -euo pipefail

mkdir -p logs

ARCH_LIST=("baseline" "deep128" "deep256")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARCH="${ARCH_LIST[${TASK_ID}]}"

SEED="${1:-42}"
TIMESTEPS="${2:-250000}"
STAGE="${3:-2C}"

echo "=============================================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}_${TASK_ID}"
echo "Node          : ${SLURM_NODENAME:-$(hostname)}"
echo "Architecture  : ${ARCH}"
echo "Seed          : ${SEED}"
echo "Stage         : ${STAGE}"
echo "Timesteps     : ${TIMESTEPS}"
echo "Start         : $(date)"
echo "=============================================================================="

# Optional cluster modules
# module load cuda/12.1 python/3.11 || true

SAVE_DIR="./models/architecture_ablation/${ARCH}_s${SEED}"
mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ "${ARCH}" == "baseline" && -f "models/curriculum_stage2/stage2/stage2_final.zip" ]]; then
    EXTRA_ARGS+=("--checkpoint" "models/curriculum_stage2/stage2/stage2_final.zip")
fi

bash scripts/train_hpc.sh \
    --stage "${STAGE}" \
    --architecture "${ARCH}" \
    --save-dir "${SAVE_DIR}" \
    --total-timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --n-envs 8 \
    --eval-freq 5000 \
    "${EXTRA_ARGS[@]}" \
    "${@:4}"

echo "End           : $(date)"
