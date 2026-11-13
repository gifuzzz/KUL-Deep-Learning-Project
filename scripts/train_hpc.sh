#!/usr/bin/env bash
# ==============================================================================
# FlameBox HPC Training Entrypoint Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=============================================================================="
echo "Starting FlameBox HPC Training"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: ${PROJECT_ROOT}"
echo "=============================================================================="

# Detect Python interpreter
PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/../fire-rescue-project-1.0/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/../fire-rescue-project-1.0/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: No suitable Python executable found!" >&2
    exit 1
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
if "${PYTHON_BIN}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA device: $("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0))')"
    DEFAULT_DEVICE="cuda"
else
    echo "CUDA not available; defaulting to CPU."
    DEFAULT_DEVICE="cpu"
fi

# Ensure output directories exist
mkdir -p logs models/curriculum

# Determine device argument: pass through user arguments or default to detected device
DEVICE_ARG=()
if [[ "$*" != *"--device"* ]]; then
    DEVICE_ARG=("--device" "${DEFAULT_DEVICE}")
fi

echo "Launching training pipeline..."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" training/train_ppo.py "${DEVICE_ARG[@]}" "$@"
