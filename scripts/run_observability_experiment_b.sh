#!/usr/bin/env bash
# ==============================================================================
# Phase 3 Production Run: Experiment B (Observability Curriculum: Full Info -> Unknown)
# Total Timesteps: 1,000,000 steps
# Architecture: baseline (1.43M params)
# Observability Schedule:
#   B1: 0 - 200,000 steps (100% geometry + victim known)
#   B2: 200,000 - 400,000 steps (100% geometry + victim hidden)
#   B3a: 400,000 - 550,000 steps (75% geometry, victim hidden)
#   B3b: 550,000 - 700,000 steps (50% geometry, victim hidden)
#   B3c: 700,000 - 800,000 steps (25% geometry, victim hidden)
#   B4: 800,000 - 1,000,000 steps (0% geometry, fully unknown)
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
SAVE_DIR="models/observability_ablation/experiment_b_s${SEED}"
TOTAL_TIMESTEPS=1000000

echo "=============================================================================="
echo "Phase 3: Launching Experiment B (Observability Curriculum) 1,000,000 steps"
echo "Seed         : ${SEED}"
echo "Device       : ${DEVICE}"
echo "Save Dir     : ${SAVE_DIR}"
echo "Start Time   : $(date)"
echo "=============================================================================="

mkdir -p "${SAVE_DIR}" logs

"${PYTHON_BIN}" training/train_ppo.py \
    --observability-mode B \
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
echo "Experiment B production run completed."
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
