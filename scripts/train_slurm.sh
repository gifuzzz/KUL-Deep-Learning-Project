#!/usr/bin/env bash
#SBATCH --job-name=flamebox_ppo
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ==============================================================================
# FlameBox Slurm Batch Submission Script
# ==============================================================================
set -euo pipefail

mkdir -p logs

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node  : ${SLURM_NODENAME:-$(hostname)}"
echo "Start : $(date)"

# Optional environment module loads (uncomment and adapt if cluster requires)
# module purge || true
# module load cuda/12.1 python/3.11 || true

# Execute training via HPC entrypoint
bash scripts/train_hpc.sh "$@"

echo "End   : $(date)"
