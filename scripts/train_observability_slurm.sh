#!/usr/bin/env bash
#SBATCH --job-name=obs_curriculum
#SBATCH --output=logs/slurm_obs_%j.out
#SBATCH --error=logs/slurm_obs_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ==============================================================================
# FlameBox HPC Slurm Submission Script for Observability Curriculum (Phase 3)
#
# Usage:
#   sbatch scripts/train_observability_slurm.sh A 42   # Run Experiment A with seed 42
#   sbatch scripts/train_observability_slurm.sh B 42   # Run Experiment B with seed 42
# ==============================================================================
set -euo pipefail

MODE="${1:-B}"
SEED="${2:-42}"

mkdir -p logs

echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : ${SLURM_NODENAME:-$(hostname)}"
echo "Mode   : ${MODE}"
echo "Seed   : ${SEED}"
echo "Start  : $(date)"

if [[ "${MODE}" == "A" || "${MODE}" == "a" ]]; then
    bash scripts/run_observability_experiment_a.sh "${SEED}"
elif [[ "${MODE}" == "B" || "${MODE}" == "b" ]]; then
    bash scripts/run_observability_experiment_b.sh "${SEED}"
else
    echo "ERROR: Unknown observability mode '${MODE}'. Choose 'A' or 'B'." >&2
    exit 1
fi

echo "End    : $(date)"
