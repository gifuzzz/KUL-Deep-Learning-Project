#!/usr/bin/env bash
# ==============================================================================
# Phase 6 Production Run: Spawn-Relative Distance Curriculum & Long-Distance Rescue
# Continues from the validated 50k pilot checkpoint:
# models/phase6_spawn_curriculum/pilot_lr1e4_50k/final_model.zip
#
# Tasks:
#   1. Full Rescue (40%): Unknown multi-room building with spawn-relative distance ladder
#   2. Return Training (35%): R1..R4 Route-Complexity with consistent branching memory
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

DEVICE="mps"
if "${PYTHON_BIN}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
elif ! "${PYTHON_BIN}" -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="cpu"
fi

START_CHECKPOINT="models/phase6_spawn_curriculum/pilot_lr1e4_50k/final_model.zip"
SAVE_DIR="models/phase6_spawn_curriculum/production_250k"
TOTAL_TIMESTEPS=250000

echo "=============================================================================="
echo "Phase 6 Production Run: Spawn-Relative Distance Curriculum"
echo "Starting Model: ${START_CHECKPOINT}"
echo "Device        : ${DEVICE}"
echo "Save Dir      : ${SAVE_DIR}"
echo "Budget        : ${TOTAL_TIMESTEPS} additional steps"
echo "Task Mixture  : 40% Full (Spawn Ladder), 35% Route-Complexity Return, 25% Discovery"
echo "Start Time    : $(date)"
echo "=============================================================================="

mkdir -p "${SAVE_DIR}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --stage 3 \
    --checkpoint "${START_CHECKPOINT}" \
    --initial-substage "3C" \
    --multi-task \
    --task-mixture "0.40,0.35,0.25" \
    --return-mode "route_complexity" \
    --discovery-reward 2.0 \
    --geometry-mode "edge" \
    --learning-rate 0.0001 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 25000 \
    --checkpoint-freq 50000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --device "${DEVICE}" \
    --seed 42 \
    --save-dir "${SAVE_DIR}"

FINAL_CKPT="${SAVE_DIR}/final_model.zip"
if [[ ! -f "${FINAL_CKPT}" ]]; then
    FINAL_CKPT="${SAVE_DIR}/stage3_spawn_curriculum/stage3_final.zip"
fi

echo "Production training complete at $(date). Saved to ${FINAL_CKPT}"
echo "Running 200-episode matched spawn-ladder benchmark..."

"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier spawn \
    --geometry-mode edge \
    --workers 12 \
    --output-json "${SAVE_DIR}/eval200_spawn_final.json"

echo "Running 200-episode matched exit-ladder benchmark..."

"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --geometry-mode edge \
    --workers 12 \
    --output-json "${SAVE_DIR}/eval200_exit_final.json"

echo "Recording representative trajectories for Visual Debugger..."
"${PYTHON_BIN}" scripts/record_representative_episodes.py --model "${FINAL_CKPT}"

echo "=============================================================================="
echo "Phase 6 Production Benchmark Complete! Results saved to ${SAVE_DIR}/"
echo "=============================================================================="
