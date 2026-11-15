#!/usr/bin/env bash
# ==============================================================================
# Multi-Task Reinforcement Learning Curriculum: 250,000-Step Production Run
# Tasks:
#   1. Full Rescue (40%): Standard unknown building, native rescue objective
#   2. Return Training (35%): Carried victim deep in building (65% at d>=10)
#   3. Discovery Training (25%): Distant hidden victim, terminates on sighting (+2.0 reward)
# Starting Checkpoint: models/observability_ablation/experiment_b_s42/final_model.zip
# Target: Convert surging distant discovery and pickup into complete distant rescues.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Detect Python interpreter
PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/../fire-rescue-project-1.0/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/../fire-rescue-project-1.0/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

DEVICE="cpu"
if "${PYTHON_BIN}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
elif "${PYTHON_BIN}" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
fi

SEED="${1:-42}"
if [[ $# -gt 0 ]]; then
    shift
fi

START_CHECKPOINT="models/observability_ablation/experiment_b_s42/final_model.zip"
SAVE_DIR="models/multitask_curriculum/production_250k_s${SEED}"
TOTAL_TIMESTEPS=250000

echo "=============================================================================="
echo "Multi-Task RL Curriculum: Production Run (${TOTAL_TIMESTEPS} steps)"
echo "Starting Model: ${START_CHECKPOINT}"
echo "Seed          : ${SEED}"
echo "Device        : ${DEVICE}"
echo "Save Dir      : ${SAVE_DIR}"
echo "Task Mixture  : 40% Full, 35% Return, 25% Discovery"
echo "Start Time    : $(date)"
echo "=============================================================================="

mkdir -p "${SAVE_DIR}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.4,0.35,0.25" \
    --discovery-reward 2.0 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 25000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --save-dir "${SAVE_DIR}" \
    --stage2c-mode retention_aware \
    "$@"

echo "Production training completed at $(date). Saved to ${SAVE_DIR}/final_model.zip"
echo "Now running 200-episode matched unshaped evaluation across all 6 tiers..."

FINAL_CKPT="${SAVE_DIR}/final_model.zip"
if [[ ! -f "${FINAL_CKPT}" ]]; then
    FINAL_CKPT="${SAVE_DIR}/stage2c/stage2c_final.zip"
fi

"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "${DEVICE}" \
    --output-json "${SAVE_DIR}/eval200_final.json"

echo "=============================================================================="
echo "Multi-Task Production Benchmark Complete! Results saved to ${SAVE_DIR}/eval200_final.json"
echo "=============================================================================="
