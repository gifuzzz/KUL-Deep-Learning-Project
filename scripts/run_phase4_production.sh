#!/usr/bin/env bash
# ==============================================================================
# Phase 4 Production Run: Route-Complexity Return Curriculum (200,000 steps)
# Continues from the validated 50k pilot checkpoint:
# models/phase4_return_curriculum/pilot_route_complexity_50k/final_model.zip
#
# Tasks:
#   1. Full Rescue (40%): Standard unknown building, native rescue objective
#   2. Return Training (35%): R1..R4 Route-Complexity with physically consistent branching memory
#   3. Discovery Training (25%): Distant hidden victim exploration
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
fi

START_CHECKPOINT="models/phase4_return_curriculum/pilot_route_complexity_50k/final_model.zip"
SAVE_DIR="models/phase4_return_curriculum/production_route_complexity_250k"
TOTAL_TIMESTEPS=200000

echo "=============================================================================="
echo "Phase 4 Production Run: Route-Complexity Return Curriculum"
echo "Starting Model: ${START_CHECKPOINT}"
echo "Device        : ${DEVICE}"
echo "Save Dir      : ${SAVE_DIR}"
echo "Budget        : ${TOTAL_TIMESTEPS} additional steps"
echo "Task Mixture  : 40% Full, 35% Route-Complexity Return (R1..R4), 25% Discovery"
echo "Start Time    : $(date)"
echo "=============================================================================="

mkdir -p "${SAVE_DIR}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.40,0.35,0.25" \
    --return-mode "route_complexity" \
    --discovery-reward 2.0 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 25000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --device "${DEVICE}" \
    --seed 42 \
    --save-dir "${SAVE_DIR}" \
    --stage2c-mode retention_aware

FINAL_CKPT="${SAVE_DIR}/final_model.zip"
if [[ ! -f "${FINAL_CKPT}" ]]; then
    FINAL_CKPT="${SAVE_DIR}/stage2c/stage2c_final.zip"
fi

echo "Production training complete at $(date). Saved to ${FINAL_CKPT}"
echo "Running 200-episode matched 6-tier benchmark (seeds 50000..50199)..."

"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "cpu" \
    --output-json "${SAVE_DIR}/eval200_final.json"

echo "Running Distant Failure Diagnosis on Production Checkpoint..."
"${PYTHON_BIN}" scripts/diagnose_distant_failures.py \
    --model "${FINAL_CKPT}" \
    --episodes 200 \
    --seed-start 50000 \
    --device "cpu" \
    --out-dir "${SAVE_DIR}/distant_diagnosis"

echo "=============================================================================="
echo "Phase 4 Production Benchmark Complete! Results saved to ${SAVE_DIR}/"
echo "=============================================================================="
