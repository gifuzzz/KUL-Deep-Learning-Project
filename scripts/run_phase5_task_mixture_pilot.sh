#!/usr/bin/env bash
# ==============================================================================
# Phase 5 Controlled Task-Mixture Pilot Experiments
# Comparing three balanced multi-task training mixtures (50,000 steps each):
#   Variant A (Current mixture): 40% Full, 35% Return, 25% Discovery
#   Variant B (More full rescue): 55% Full, 25% Return, 20% Discovery
#   Variant C (More discovery):   40% Full, 25% Return, 35% Discovery
#
# Starting Checkpoint:
#   models/phase4_return_curriculum/production_route_complexity_250k/final_model.zip
#
# All variants:
#   - Start from identical Phase 4 production weights.
#   - Use identical PPO hyperparameters (lr=1e-4, 8 envs, n_steps=128).
#   - Retain route-complexity return training (R1..R4 with branching memory).
#   - Evaluate on identical 200-episode matched 6-tier benchmark (seeds 50000..50199).
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
START_CHECKPOINT="models/phase4_return_curriculum/production_route_complexity_250k/final_model.zip"
TOTAL_TIMESTEPS=50000

echo "=============================================================================="
echo "Phase 5 Controlled Task-Mixture Pilot Experiments"
echo "Starting Model: ${START_CHECKPOINT}"
echo "Device        : ${DEVICE}"
echo "Budget        : ${TOTAL_TIMESTEPS} steps per variant"
echo "Variants      : A (40/35/25), B (55/25/20), C (40/25/35)"
echo "Start Time    : $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. VARIANT A: CURRENT MIXTURE (40% Full, 35% Return, 25% Discovery)
# ------------------------------------------------------------------------------
SAVE_DIR_A="models/phase5_balanced_multitask/pilot_variant_a_40_35_25"
echo ""
echo ">>> [1/3] Launching Variant A: Current Mixture (40% Full, 35% Return, 25% Discovery)..."
mkdir -p "${SAVE_DIR_A}" logs

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
# 2. VARIANT B: MORE FULL RESCUE (55% Full, 25% Return, 20% Discovery)
# ------------------------------------------------------------------------------
SAVE_DIR_B="models/phase5_balanced_multitask/pilot_variant_b_55_25_20"
echo ""
echo ">>> [2/3] Launching Variant B: More Full Rescue (55% Full, 25% Return, 20% Discovery)..."
mkdir -p "${SAVE_DIR_B}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.55,0.25,0.20" \
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

# ------------------------------------------------------------------------------
# 3. VARIANT C: MORE DISCOVERY (40% Full, 25% Return, 35% Discovery)
# ------------------------------------------------------------------------------
SAVE_DIR_C="models/phase5_balanced_multitask/pilot_variant_c_40_25_35"
echo ""
echo ">>> [3/3] Launching Variant C: More Discovery (40% Full, 25% Return, 35% Discovery)..."
mkdir -p "${SAVE_DIR_C}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --checkpoint "${START_CHECKPOINT}" \
    --stage 2C \
    --initial-substage 2C_distant \
    --multi-task \
    --task-mixture "0.40,0.25,0.35" \
    --return-mode "route_complexity" \
    --discovery-reward 2.0 \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-freq 10000 \
    --n-eval-episodes 20 \
    --n-envs 8 \
    --learning-rate 0.0001 \
    --device "${DEVICE}" \
    --seed 42 \
    --save-dir "${SAVE_DIR_C}" \
    --stage2c-mode retention_aware

FINAL_CKPT_C="${SAVE_DIR_C}/final_model.zip"
if [[ ! -f "${FINAL_CKPT_C}" ]]; then
    FINAL_CKPT_C="${SAVE_DIR_C}/stage2c/stage2c_final.zip"
fi

echo "Variant C Training Complete. Evaluating on 200-episode matched 6-tier benchmark..."
"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${FINAL_CKPT_C}" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "cpu" \
    --output-json "${SAVE_DIR_C}/eval200_final.json"

echo "Running Distant Failure Diagnosis on Variant C..."
"${PYTHON_BIN}" scripts/diagnose_distant_failures.py \
    --model "${FINAL_CKPT_C}" \
    --episodes 200 \
    --seed-start 50000 \
    --device "cpu" \
    --out-dir "${SAVE_DIR_C}/distant_diagnosis"

echo "=============================================================================="
echo "Phase 5 Controlled Pilot Runs Complete!"
echo "Variant A (40/35/25): ${FINAL_CKPT_A}"
echo "Variant B (55/25/20): ${FINAL_CKPT_B}"
echo "Variant C (40/25/35): ${FINAL_CKPT_C}"
echo "Finished at: $(date)"
echo "=============================================================================="
