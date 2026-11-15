#!/usr/bin/env bash
# ==============================================================================
# Phase 3 Production Run: Experiment A (Tabula Rasa / Unknown from Step 0)
# Total Timesteps: 1,000,000 steps
# Architecture: baseline (1.43M params)
# Observability Mode: A (Standard partial observability from step 0)
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

SEED="${1:-42}"
SAVE_DIR="models/observability_ablation/experiment_a_s${SEED}"
TOTAL_TIMESTEPS=1000000

echo "=============================================================================="
echo "Phase 3: Launching Experiment A (Tabula Rasa) 1,000,000 steps"
echo "Seed         : ${SEED}"
echo "Device       : ${DEVICE}"
echo "Save Dir     : ${SAVE_DIR}"
echo "Start Time   : $(date)"
echo "=============================================================================="

mkdir -p "${SAVE_DIR}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --observability-mode A \
    --stage 2C \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --save-dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --eval-freq 25000 \
    --checkpoint-freq 100000 \
    --device "${DEVICE}" \
    --architecture baseline \
    "$@"

echo "=============================================================================="
echo "Experiment A production run completed."
echo "Running unshaped held-out evaluation across all distance tiers..."
echo "=============================================================================="

"${PYTHON_BIN}" scripts/diagnose_curriculum.py \
    --model "${SAVE_DIR}/final_model.zip" \
    --episodes 200 \
    --seed-start 50000 \
    --tier all \
    --device "${DEVICE}" \
    --output-json "${SAVE_DIR}/eval200_final.json"

echo "End Time: $(date)"
