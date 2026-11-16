#!/usr/bin/env bash
# ==============================================================================
# Phase 4 Controlled Pilot Experiment: Long-Distance Return Navigation
# Comparing:
#   Variant A (Control): Legacy single-corridor return training
#   Variant B (Improved): Route-Complexity return curriculum (R1..R4 with branching memory)
#
# Starting Checkpoint: models/multitask_curriculum/production_250k_s42/final_model.zip
# Task Mixture: 40% Full Rescue, 35% Return Training, 25% Discovery Training
# Budget: 50,000 environment steps per variant
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

START_CHECKPOINT="models/multitask_curriculum/production_250k_s42/final_model.zip"
TOTAL_TIMESTEPS=50000

echo "=============================================================================="
echo "Phase 4 Controlled Pilot: Return-Navigation Curriculum Comparison"
echo "Starting Model: ${START_CHECKPOINT}"
echo "Device        : ${DEVICE}"
echo "Budget        : ${TOTAL_TIMESTEPS} steps per variant"
echo "Task Mixture  : 40% Full Rescue, 35% Return Training, 25% Discovery Training"
echo "Start Time    : $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. VARIANT A: CONTROL (Legacy Single-Corridor Return)
# ------------------------------------------------------------------------------
SAVE_DIR_A="models/phase4_return_curriculum/pilot_control_50k"
echo ""
echo ">>> [1/2] Launching Variant A: Control (Legacy Return)..."
mkdir -p "${SAVE_DIR_A}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.40,0.35,0.25" \
    --return-mode "legacy" \
    --discovery-reward 2.0 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --device "${DEVICE}" \
    --seed 42 \
    --save-dir "${SAVE_DIR_A}" \
    --stage2c-mode retention_aware

FINAL_CKPT_A="${SAVE_DIR_A}/final_model.zip"
if [[ ! -f "${FINAL_CKPT_A}" ]]; then
    FINAL_CKPT_A="${SAVE_DIR_A}/stage2c/stage2c_final.zip"
fi

echo "Variant A Training Complete. Evaluating on 200-episode matched 6-tier benchmark..."
"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT_A}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "cpu" \
    --output-json "${SAVE_DIR_A}/eval200_final.json"

echo "Running Distant Failure Diagnosis on Variant A..."
"${PYTHON_BIN}" scripts/diagnose_distant_failures.py \
    --model "${FINAL_CKPT_A}" \
    --episodes 200 \
    --seed-start 50000 \
    --device "cpu" \
    --out-dir "${SAVE_DIR_A}/distant_diagnosis"

# ------------------------------------------------------------------------------
# 2. VARIANT B: ROUTE-COMPLEXITY RETURN CURRICULUM (R1..R4 with Branching Memory)
# ------------------------------------------------------------------------------
SAVE_DIR_B="models/phase4_return_curriculum/pilot_route_complexity_50k"
echo ""
echo ">>> [2/2] Launching Variant B: Improved Route-Complexity Return Curriculum..."
mkdir -p "${SAVE_DIR_B}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.40,0.35,0.25" \
    --return-mode "route_complexity" \
    --discovery-reward 2.0 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --device "${DEVICE}" \
    --seed 42 \
    --save-dir "${SAVE_DIR_B}" \
    --stage2c-mode retention_aware

FINAL_CKPT_B="${SAVE_DIR_B}/final_model.zip"
if [[ ! -f "${FINAL_CKPT_B}" ]]; then
    FINAL_CKPT_B="${SAVE_DIR_B}/stage2c/stage2c_final.zip"
fi

echo "Variant B Training Complete. Evaluating on 200-episode matched 6-tier benchmark..."
"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT_B}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "cpu" \
    --output-json "${SAVE_DIR_B}/eval200_final.json"

echo "Running Distant Failure Diagnosis on Variant B..."
"${PYTHON_BIN}" scripts/diagnose_distant_failures.py \
    --model "${FINAL_CKPT_B}" \
    --episodes 200 \
    --seed-start 50000 \
    --device "cpu" \
    --out-dir "${SAVE_DIR_B}/distant_diagnosis"

echo "=============================================================================="
echo "Phase 4 Controlled Pilot Run Complete!"
echo "Variant A (Control): ${FINAL_CKPT_A}"
echo "Variant B (Route-Complexity): ${FINAL_CKPT_B}"
echo "Completed at: $(date)"
echo "=============================================================================="
